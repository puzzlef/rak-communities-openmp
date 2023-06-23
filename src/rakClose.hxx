#pragma once
#include <utility>
#include <vector>
#include "_main.hxx"
#include "update.hxx"
#include "rakClose.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using std::pair;
using std::tuple;
using std::vector;
using std::make_pair;
using std::swap;
using std::move;
using std::get;




// RAK HASHTABLES
// --------------

/**
 * Allocate a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param S size of each hashtable
 */
template <class K, class W>
inline void rakCloseAllocateHashtables(vector<vector<K>>& vcs, vector<vector<W>>& vcout, size_t S) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i]   = vector<K>();
    vcout[i] = vector<W>(S);
  }
}


/**
 * Free a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 */
template <class K, class W>
inline void rakCloseFreeHashtables(vector<vector<K>>& vcs, vector<vector<W>>& vcout) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i].clear();
    vcout[i].clear();
  }
}




// RAK INITIALIZE
// --------------

/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (updated)
 * @param x original graph
 */
template <class G, class K>
inline void rakCloseInitialize(vector<K>& vcom, const G& x) {
  x.forEachVertexKey([&](auto u) { vcom[u] = u; });
}

#ifdef OPENMP
template <class G, class K>
inline void rakCloseInitializeOmp(vector<K>& vcom, const G& x) {
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    vcom[u] = u;
  }
}
#endif


/**
 * Initialize communities from given initial communities.
 * @param vcom community each vertex belongs to (updated)
 * @param x original graph
 * @param q initial community each vertex belongs to
 */
template <class G, class K>
inline void rakCloseInitializeFrom(vector<K>& vcom, const G& x, const vector<K>& q) {
  x.forEachVertexKey([&](auto u) { vcom[u] = q[u]; });
}

#ifdef OPENMP
template <class G, class K>
inline void rakCloseInitializeFromOmp(vector<K>& vcom, const G& x, const vector<K>& q) {
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    vcom[u] = q[u];
  }
}
#endif




// RAK CHOOSE COMMUNITY
// --------------------

/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class K, class V, class W>
inline void rakCloseScanCommunity(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom) {
  if (!SELF && u==v) return;
  K c = vcom[v];
  if (!vcout[c]) vcs.push_back(c);
  vcout[c] += w;
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class G, class K, class W>
inline void rakCloseScanCommunities(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) { rakCloseScanCommunity<SELF>(vcs, vcout, u, v, w, vcom); });
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (updated)
 * @param vcout communities vertex u is linked to (updated)
 */
template <class K, class W>
inline void rakCloseClearScan(vector<K>& vcs, vector<W>& vcout) {
  for (K c : vcs)
    vcout[c] = W();
  vcs.clear();
}


/**
 * Choose connected community with most weight.
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param vcs communities vertex u is linked to
 * @param vcout total edge weight from vertex u to community C
 * @returns [best community, best edge weight to community]
 */
template <class G, class K, class W>
inline pair<K, W> rakCloseChooseCommunity(const G& x, K u, const vector<K>& vcom, const vector<K>& vcs, const vector<W>& vcout) {
  K d = vcom[u];
  K cmax = K();
  W wmax = W();
  for (K c : vcs)
    if (vcout[c]>wmax) { cmax = c; wmax = vcout[c]; }
  return make_pair(cmax, wmax);
}




// RAK MOVE ITERATION
// ------------------

/**
 * Move each vertex to its best community.
 * @param vcom community each vertex belongs to (updated)
 * @param vaff is vertex affected (updated)
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @returns number of changed vertices
 */
template <class G, class K, class W, class B>
inline size_t rakCloseMoveIteration(vector<K>& vcom, vector<B>& vaff, vector<K>& vcs, vector<W>& vcout, const G& x) {
  size_t a = 0;
  x.forEachVertexKey([&](auto u) {
    if (!vaff[u]) return;
    K d = vcom[u];
    rakCloseClearScan(vcs, vcout);
    rakCloseScanCommunities(vcs, vcout, x, u, vcom);
    auto [c, w] = rakCloseChooseCommunity(x, u, vcom, vcs, vcout);
    if (c && c!=d) { vcom[u] = c; ++a; x.forEachEdgeKey(u, [&](auto v) { vaff[v] = B(1); }); }
    vaff[u] = B(0);
  });
  return a;
}

#ifdef OPENMP
template <class G, class K, class W, class B>
inline size_t rakCloseMoveIterationOmp(vector<K>& vcom, vector<B>& vaff, vector<vector<K>>& vcs, vector<vector<W>>& vcout, const G& x) {
  size_t a = K();
  size_t S = x.span();
  #pragma omp parallel for schedule(dynamic, 2048) reduction(+:a)
  for (K u=0; u<S; ++u) {
    int t = omp_get_thread_num();
    if (!x.hasVertex(u)) continue;
    if (!vaff[u]) continue;
    K d = vcom[u];
    rakCloseClearScan(vcs[t], vcout[t]);
    rakCloseScanCommunities(vcs[t], vcout[t], x, u, vcom);
    auto [c, w] = rakCloseChooseCommunity(x, u, vcom, vcs[t], vcout[t]);
    if (c && c!=d) { vcom[u] = c; ++a; x.forEachEdgeKey(u, [&](auto v) { vaff[v] = B(1); }); }
    vaff[u] = B(0);
  }
  return a;
}
#endif




// RAK
// ---

template <class FLAG=char, class G, class K, class FM>
RakResult<K> rakCloseSeq(const G& x, const vector<K>* q, const RakOptions& o, FM fm) {
  using V = typename G::edge_value_type;
  using W = RAK_WEIGHT_TYPE;
  using B = FLAG;
  int l = 0;
  size_t S = x.span();
  size_t N = x.order();
  vector<K> vcom(S), vcs;
  vector<W> vcout(S);
  vector<B> vaff(S);
  float tm = 0;
  float t  = measureDuration([&]() {
    tm += measureDuration([&]() { fm(vaff); });
    if (q) rakCloseInitializeFrom(vcom, x, *q);
    else   rakCloseInitialize(vcom, x);
    for (l=0; l<o.maxIterations;) {
      size_t n = rakCloseMoveIteration(vcom, vaff, vcs, vcout, x); ++l;
      if (double(n)/N <= o.tolerance) break;
    }
  }, o.repeat);
  return {vcom, l, t, tm/o.repeat};
}


#ifdef OPENMP
template <class FLAG=char, class G, class K, class FM>
RakResult<K> rakCloseOmp(const G& x, const vector<K>* q, const RakOptions& o, FM fm) {
  using V = typename G::edge_value_type;
  using W = RAK_WEIGHT_TYPE;
  using B = FLAG;
  int l = 0;
  int T = omp_get_max_threads();
  size_t S = x.span();
  size_t N = x.order();
  vector<K> vcom(S);
  vector<B> vaff(S);
  vector<vector<K>> vcs(T);
  vector<vector<W>> vcout(T);
  rakCloseAllocateHashtables(vcs, vcout, S);
  float tm = 0;
  float t  = measureDuration([&]() {
    tm += measureDuration([&]() { fm(vaff); });
    if (q) rakCloseInitializeFromOmp(vcom, x, *q);
    else   rakCloseInitializeOmp(vcom, x);
    for (l=0; l<o.maxIterations;) {
      size_t n = rakCloseMoveIterationOmp(vcom, vaff, vcs, vcout, x); ++l;
      if (double(n)/N <= o.tolerance) break;
    }
  }, o.repeat);
  rakCloseFreeHashtables(vcs, vcout);
  return {vcom, l, t, tm/o.repeat};
}
#endif




// RAK STATIC
// ----------

template <class FLAG=char, class G, class K>
inline RakResult<K> rakCloseStaticSeq(const G& x, const vector<K>* q=nullptr, const RakOptions& o={}) {
  auto fm = [](auto& vaff) { fillValueU(vaff, FLAG(1)); };
  return rakCloseSeq<FLAG>(x, q, o, fm);
}

#ifdef OPENMP
template <class FLAG=char, class G, class K>
inline RakResult<K> rakCloseStaticOmp(const G& x, const vector<K>* q=nullptr, const RakOptions& o={}) {
  auto fm = [](auto& vaff) { fillValueU(vaff, FLAG(1)); };
  return rakCloseOmp<FLAG>(x, q, o, fm);
}
#endif
