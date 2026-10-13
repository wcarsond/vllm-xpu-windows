// one-process-allreduce.cpp - the decode all_reduce as a one-process executor would run it: both B60s in one sycl::context,
// each card writes its half as tagged words into one shared buffer and reads the other's, at the same time (an
// all_reduce, not a ping-pong). Which store width and how many work-groups make it fastest, and does peer access exist?
// Variants:
//   W   bytes per store: 8 (one tagged word, as the exchange does today), 32 (four words), 64 (eight words). Every word
//       keeps its own tag, so a wide store that lands in pieces is still caught by the reader.
//   G   work-groups per card (each takes its own slice of the words, no cross-group sync needed)
//   where: host (one malloc_host buffer, both cards over PCIe) or peer (the writer stores straight into the READER's card
//       memory; only if the driver reports peer access)
// Modes: persistent (one kernel runs every round: pure latency) and launched (one fused write-wait-add kernel per round on
// each card, enqueued from one host thread, card 0 then card 1: what eager and graph replay will do).
// Usage:  one-process-allreduce.exe <words> <rounds>          (run by one-process-allreduce.input.cmd)
// Why (2026-10-13): the ping-pong probe gave 14.3 us one way for 896 words, rising ~14 ns a word (4096 words: 57.8 us),
// so the 7B's real decode all_reduce (1,792 tagged words = 7 KB of data) would cost ~27 us, no better than the two-process
// exchange's 28 us. The suspect is the writer's 8-byte uncached stores (one PCIe write each); if wide stores cut that,
// the one-process executor is worth building.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using wprops = decltype(oex::properties(
    iex::write_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L3>>));
using rprops = decltype(oex::properties(
    iex::read_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L2, oex::cache_level::L3>>));

static inline uint64_t tagged(uint32_t d, uint32_t t) { return (uint64_t)d | ((uint64_t)t << 32); }
constexpr int WG = 512;
constexpr uint32_t SPIN = 6000000;

// data word i of card c in round s: something both sides can recompute
static inline uint32_t val(int c, size_t i, uint32_t s) { return (uint32_t)(i * (c ? 11u : 7u) + s); }

// one round's work for one lane: write my vectors of the slice, then wait for the peer's and check them
template <int V>
static inline void round_body(size_t lid, size_t v0, size_t v1, int me, uint32_t s, uint64_t* out, const uint64_t* in,
                              uint32_t* stat) {
  using vec = sycl::vec<uint64_t, V>;
  oex::annotated_ptr<vec, wprops> wp{reinterpret_cast<vec*>(out)};
  oex::annotated_ptr<const vec, rprops> rp{reinterpret_cast<const vec*>(in)};
  for (size_t v = v0 + lid; v < v1; v += WG) {
    vec w;
    for (int k = 0; k < V; ++k) w[k] = tagged(val(me, v * V + k, s), s);
    wp[v] = w;
  }
  for (size_t v = v0 + lid; v < v1; v += WG) {
    vec r; uint32_t n = 0;
    for (;;) {
      { vec t = rp[v]; r = t; }
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
      bool ok = true;
      for (int k = 0; k < V; ++k) ok = ok && (uint32_t)(r[k] >> 32) == s;
      if (ok) break;
      if (++n > SPIN) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> a(stat[2 + me]); a++; break; }
    }
    for (int k = 0; k < V; ++k)
      if ((uint32_t)r[k] != val(1 - me, v * V + k, s)) { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> a(stat[me]); a++; }
  }
}

struct Bufs { uint64_t* to[2][2]; };   // to[c][parity]: the words card c READS (written by card 1-c)

template <int V>
static double persistent(sycl::queue* q, Bufs b, size_t words, int rounds, int groups, uint32_t* stat) {
  const size_t nvec = words / V;
  auto t0 = std::chrono::steady_clock::now();
  sycl::event e[2];
  for (int c = 1; c >= 0; --c) {
    uint64_t* o0 = b.to[1 - c][0]; uint64_t* o1 = b.to[1 - c][1];
    const uint64_t* i0 = b.to[c][0]; const uint64_t* i1 = b.to[c][1];
    e[c] = q[c].parallel_for(sycl::nd_range<1>(WG * groups, WG), [=](sycl::nd_item<1> it) {
      const size_t g = it.get_group(0), lid = it.get_local_id(0);
      const size_t per = (nvec + groups - 1) / groups, v0 = g * per, v1 = v0 + per < nvec ? v0 + per : nvec;
      for (int s = 1; s <= rounds; ++s) {
        round_body<V>(lid, v0, v1, c, (uint32_t)s, (s & 1) ? o1 : o0, (s & 1) ? i1 : i0, stat);
        sycl::group_barrier(it.get_group());
      }
    });
  }
  e[0].wait(); e[1].wait();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6 / rounds;
}

template <int V>
static double launched(sycl::queue* q, Bufs b, size_t words, int rounds, int groups, uint32_t* stat, uint32_t base) {
  const size_t nvec = words / V;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 1; r <= rounds; ++r) {
    const uint32_t s = base + (uint32_t)r;
    for (int c = 0; c < 2; ++c) {
      uint64_t* o = b.to[1 - c][s & 1];
      const uint64_t* in = b.to[c][s & 1];
      q[c].parallel_for(sycl::nd_range<1>(WG * groups, WG), [=](sycl::nd_item<1> it) {
        const size_t g = it.get_group(0), lid = it.get_local_id(0);
        const size_t per = (nvec + groups - 1) / groups, v0 = g * per, v1 = v0 + per < nvec ? v0 + per : nvec;
        round_body<V>(lid, v0, v1, c, s, o, in, stat);
      });
    }
  }
  q[0].wait(); q[1].wait();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6 / rounds;
}

static double empty_launch(sycl::queue* q, int rounds) {
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < rounds; ++r)
    for (int c = 0; c < 2; ++c) q[c].parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1>) {});
  q[0].wait(); q[1].wait();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6 / rounds;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const size_t words = argc > 1 ? std::atoll(argv[1]) : 1792;
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 2000;
  std::vector<sycl::device> cards;
  for (auto& p : sycl::platform::get_platforms())
    if (p.get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos)
      for (auto& d : p.get_devices())
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("B60") != std::string::npos) cards.push_back(d);
  if (cards.size() < 2) { std::printf("need two B60s, found %zu\n", cards.size()); return 3; }
  sycl::context ctx({cards[0], cards[1]});
  sycl::queue q[2] = {sycl::queue(ctx, cards[0], sycl::property::queue::in_order{}),
                      sycl::queue(ctx, cards[1], sycl::property::queue::in_order{})};
  bool p01 = false, p10 = false;
  try {
    p01 = cards[0].ext_oneapi_can_access_peer(cards[1], sycl::ext::oneapi::peer_access::access_supported);
    p10 = cards[1].ext_oneapi_can_access_peer(cards[0], sycl::ext::oneapi::peer_access::access_supported);
  } catch (const sycl::exception& e) { std::printf("peer query threw: %s\n", e.what()); }
  std::printf("words %zu (%zu B tagged, %zu B data), rounds %d; peer access 0->1 %s, 1->0 %s\n", words, words * 8, words * 4,
              rounds, p01 ? "yes" : "no", p10 ? "yes" : "no");
  uint32_t* stat = sycl::malloc_host<uint32_t>(64, ctx);

  std::vector<std::string> wheres = {"host"};
  if (p01 && p10) {
    try { cards[0].ext_oneapi_enable_peer_access(cards[1]); cards[1].ext_oneapi_enable_peer_access(cards[0]); wheres.push_back("peer"); }
    catch (const sycl::exception& e) { std::printf("enable peer access threw: %s\n", e.what()); }
  }
  std::printf("empty kernel launch, both cards, one thread: %.1f us per round\n", empty_launch(q, 500));
  uint32_t base = 1000000;
  for (auto& where : wheres) {
    Bufs b;
    for (int c = 0; c < 2; ++c)
      for (int p = 0; p < 2; ++p) {
        b.to[c][p] = where == "host" ? sycl::malloc_host<uint64_t>(words, ctx) : sycl::malloc_device<uint64_t>(words, q[c]);
        q[c].memset(b.to[c][p], 0, words * 8).wait();
      }
    // 2026-10-13 16:35: 32- and 64-byte stores were 4-15x slower than 8-byte ones, and 4 groups beat 1 by ~4x: sweep the
    // group count with 8-byte stores (OP_GROUPS="1,4,8,16,32,64", OP_WIDTHS="1" in words per store)
    std::vector<int> Vs, Gs;
    { const char* e = std::getenv("OP_WIDTHS"); std::string t = e ? e : "1"; size_t a = 0; while (a < t.size()) { size_t z = t.find(',', a); if (z == std::string::npos) z = t.size(); Vs.push_back(std::atoi(t.substr(a, z - a).c_str())); a = z + 1; } }
    { const char* e = std::getenv("OP_GROUPS"); std::string t = e ? e : "1,4,8,16,32,64"; size_t a = 0; while (a < t.size()) { size_t z = t.find(',', a); if (z == std::string::npos) z = t.size(); Gs.push_back(std::atoi(t.substr(a, z - a).c_str())); a = z + 1; } }
    for (int V : Vs)
      for (int G : Gs) {
        if (words % (size_t)V) continue;
        for (int i = 0; i < 64; ++i) stat[i] = 0;
        double pe = 0, la = 0;
        try {
          // reset tags between modes: the persistent run uses tags 1..rounds, launched ones count from a fresh base
          for (int c = 0; c < 2; ++c) for (int p = 0; p < 2; ++p) q[c].memset(b.to[c][p], 0, words * 8).wait();
          pe = V == 1 ? persistent<1>(q, b, words, rounds, G, stat) : V == 4 ? persistent<4>(q, b, words, rounds, G, stat)
                                                                           : persistent<8>(q, b, words, rounds, G, stat);
          la = V == 1 ? launched<1>(q, b, words, rounds, G, stat, base) : V == 4 ? launched<4>(q, b, words, rounds, G, stat, base)
                                                                                : launched<8>(q, b, words, rounds, G, stat, base);
          base += rounds + 10;
        } catch (const sycl::exception& e) { std::printf("%s W%d G%d: SYCL exception %s\n", where.c_str(), V * 8, G, e.what()); continue; }
        std::printf("%-4s W%2d G%d: persistent %6.1f us per all_reduce, launched %6.1f us; wrong %u/%u, timeouts %u/%u\n",
                    where.c_str(), V * 8, G, pe, la, stat[0], stat[1], stat[2], stat[3]);
      }
    for (int c = 0; c < 2; ++c) for (int p = 0; p < 2; ++p) sycl::free(b.to[c][p], ctx);
  }
  sycl::free(stat, ctx);
  return 0;
}
