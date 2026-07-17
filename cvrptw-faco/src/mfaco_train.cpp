/**
 * ACO Training Module - Unified C++ implementation
 */

#include "mfaco_train.h"
#include "kd_tree.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include <omp.h>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace mfaco {

MFACO_CVRP::MFACO_CVRP(const float *coords_ptr, const float *demand_ptr,
                       int32_t n_, float capacity_, int32_t n_ants_,
                       int32_t cand_list_size, int32_t backup_list_size,
                       int32_t min_new_edges_, float decay, float alpha_,
                       float p_best_, bool use_local_search_,
                       bool disable_heuristic_, bool extend_ls_,
                       bool smooth_mmas_, int32_t fixed_steps_, bool nls_,
                       int32_t T_nls_)
    : n(n_), m(n_ - 1), n_ants(n_ants_), k(std::min(cand_list_size, n_ - 1)),
      bl(std::min(backup_list_size, std::max(0, n_ - 1 - k))),
      min_new_edges(min_new_edges_), fixed_steps(fixed_steps_), rho(decay),
      alpha(alpha_), p_best(p_best_), use_local_search(use_local_search_),
      disable_heuristic(disable_heuristic_), extend_ls(extend_ls_),
      smooth_mmas(smooth_mmas_), capacity(capacity_),
      capacity_int(static_cast<int64_t>(std::round(capacity_ * DEMAND_SCALE))),
      nls(nls_), T_nls(T_nls_), use_relocate(true), use_swap(true),
      use_2opt_star(true) {
  if (!coords_ptr || !demand_ptr) {
    throw std::runtime_error("coords_ptr and demand_ptr must not be null");
  }
  if (n < 2)
    throw std::runtime_error("n must be >= 2 (depot + at least one customer)");
  if (k > static_cast<int32_t>(MAX_CAND_LIST_SIZE)) {
    throw std::runtime_error("cand_list_size must be <= " +
                             std::to_string(MAX_CAND_LIST_SIZE));
  }
  if (capacity <= 0)
    throw std::runtime_error("capacity must be > 0");
  capacity_int = (int64_t)std::round(capacity * DEMAND_SCALE);

  coords.resize(static_cast<size_t>(n) * 2);
  std::memcpy(coords.data(), coords_ptr,
              sizeof(float) * static_cast<size_t>(n) * 2);

  demand.resize(static_cast<size_t>(n));
  std::memcpy(demand.data(), demand_ptr,
              sizeof(float) * static_cast<size_t>(n));
  demand[0] = 0.0f; // enforce

  demand_int.resize(n);
  for (int32_t i = 0; i < n; ++i) {
    demand_int[i] = (int64_t)std::round(demand[i] * DEMAND_SCALE);
  }

  build_nn_lists();
  // if (!smooth_mmas || nls)
  //   build_nn_pos();
  build_heuristic();
  build_d0();

  // source_perm.resize(m); // REMOVED
  // best_perm.resize(m);   // REMOVED
  // source_positions.assign(n, -1); // REMOVED

  // Initialize routes
  source_route.reserve(n * 2);
  best_route.reserve(n * 2);

  build_initial_solution();

  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(source_cost)
                                  : calc_trail_limits_cl(source_cost);
  tau_min = tmin;
  tau_max = tmax;
  pheromone_sparse.assign(n * k, tau_max);

  rng_.seed(42);
}

void MFACO_CVRP::seed_rng(uint64_t seed) { rng_.seed(seed); }

void MFACO_CVRP::set_time_windows(const float *windows_ptr) {
  if (!windows_ptr) {
    throw std::runtime_error("windows_ptr must not be null");
  }
  has_time_windows = true;
  ready_time.resize(n);
  due_time.resize(n);
  for (int32_t i = 0; i < n; ++i) {
    ready_time[i] = windows_ptr[i * 2];
    due_time[i] = windows_ptr[i * 2 + 1];
  }

  build_initial_solution();
  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(source_cost)
                                  : calc_trail_limits_cl(source_cost);
  tau_min = tmin;
  tau_max = tmax;
  pheromone_sparse.assign(static_cast<size_t>(n) * static_cast<size_t>(k),
                          tau_max);
}

bool MFACO_CVRP::can_append_tw(int32_t prev, int32_t node,
                               float route_time) const {
  if (!has_time_windows)
    return true;
  float arrival = route_time + dist(prev, node);
  if (arrival > due_time[node] + 1e-6f)
    return false;
  float service_start = std::max(arrival, ready_time[node]);
  return service_start + dist(node, 0) <= due_time[0] + 1e-6f;
}

bool MFACO_CVRP::route_time_feasible(const std::vector<int32_t> &route) const {
  if (!has_time_windows)
    return true;
  float route_time = 0.0f;
  int32_t prev = 0;
  for (int32_t node : route) {
    if (node == 0) {
      if (prev != 0 && route_time + dist(prev, 0) > due_time[0] + 1e-6f)
        return false;
      route_time = 0.0f;
      prev = 0;
      continue;
    }
    float arrival = route_time + dist(prev, node);
    if (arrival > due_time[node] + 1e-6f)
      return false;
    route_time = std::max(arrival, ready_time[node]);
    if (route_time + dist(node, 0) > due_time[0] + 1e-6f)
      return false;
    prev = node;
  }
  return true;
}

bool MFACO_CVRP::route_fully_feasible(const std::vector<int32_t> &route) const {
  if (route.empty() || route.front() != 0 || route.back() != 0)
    return false;

  std::vector<uint8_t> seen(n, 0);
  int32_t seen_count = 0;
  int64_t route_load = 0;
  float route_time = 0.0f;
  int32_t prev = 0;

  for (size_t i = 1; i < route.size(); ++i) {
    int32_t node = route[i];
    if (node < 0 || node >= n)
      return false;

    float arrival = route_time + dist(prev, node);
    if (node == 0) {
      if (prev != 0 && has_time_windows && arrival > due_time[0] + 1e-6f)
        return false;
      route_load = 0;
      route_time = 0.0f;
      prev = 0;
      continue;
    }

    if (seen[node])
      return false;
    seen[node] = 1;
    seen_count++;

    route_load += demand_int[node];
    if (route_load > capacity_int)
      return false;

    if (has_time_windows) {
      if (arrival > due_time[node] + 1e-6f)
        return false;
      route_time = std::max(arrival, ready_time[node]);
      if (route_time + dist(node, 0) > due_time[0] + 1e-6f)
        return false;
    } else {
      route_time = arrival;
    }
    prev = node;
  }

  return seen_count == m;
}

bool MFACO_CVRP::linked_route_feasible(
    int32_t route_id, const std::vector<int32_t> &next_node,
    const std::vector<int32_t> &node_route,
    const std::vector<int64_t> &route_loads) const {
  if (route_id < 0 || route_id >= (int32_t)route_loads.size())
    return false;
  if (route_loads[route_id] > capacity_int)
    return false;
  if (!has_time_windows)
    return true;

  int32_t depot = n + route_id;
  if (depot < 0 || depot >= (int32_t)next_node.size())
    return false;

  float route_time = 0.0f;
  int32_t prev = 0;
  int32_t w = next_node[depot];
  int32_t guard = 0;
  while (w < n) {
    if (w <= 0 || w >= n || node_route[w] != route_id)
      return false;
    float arrival = route_time + dist(prev, w);
    if (arrival > due_time[w] + 1e-6f)
      return false;
    route_time = std::max(arrival, ready_time[w]);
    if (route_time + dist(w, 0) > due_time[0] + 1e-6f)
      return false;
    prev = w;
    w = next_node[w];
    if (++guard > n)
      return false;
  }

  return route_time + dist(prev, 0) <= due_time[0] + 1e-6f;
}

bool MFACO_CVRP::linked_solution_feasible(
    int32_t num_routes, const std::vector<int32_t> &next_node,
    const std::vector<int32_t> &node_route,
    const std::vector<int64_t> &route_loads) const {
  for (int32_t r = 0; r < num_routes; ++r) {
    if (!linked_route_feasible(r, next_node, node_route, route_loads))
      return false;
  }
  return true;
}

void MFACO_CVRP::enforce_time_windows(std::vector<int32_t> &route) const {
  if (!has_time_windows || route.empty())
    return;

  std::vector<int32_t> rebuilt;
  rebuilt.reserve(route.size() + n);
  rebuilt.push_back(0);

  float route_time = 0.0f;
  int64_t route_load = 0;
  int32_t prev = 0;

  for (int32_t node : route) {
    if (node == 0)
      continue;

    bool feasible_here = can_append_tw(prev, node, route_time) &&
                         route_load + demand_int[node] <= capacity_int;
    if (!feasible_here && prev != 0) {
      rebuilt.push_back(0);
      route_time = 0.0f;
      route_load = 0;
      prev = 0;
    }

    float arrival = route_time + dist(prev, node);
    rebuilt.push_back(node);
    route_time = std::max(arrival, ready_time[node]);
    route_load += demand_int[node];
    prev = node;
  }

  if (rebuilt.back() != 0)
    rebuilt.push_back(0);
  route.swap(rebuilt);
}

void MFACO_CVRP::reset_timings() {
  time_ant = 0.0;
  time_ls = 0.0;
  time_split = 0.0;
}

// -------------------- distance --------------------
float MFACO_CVRP::dist(int32_t u, int32_t v) const {
  const size_t ou = static_cast<size_t>(u) * 2;
  const size_t ov = static_cast<size_t>(v) * 2;
  float dx = coords[ou] - coords[ov];
  float dy = coords[ou + 1] - coords[ov + 1];
  return std::sqrt(dx * dx + dy * dy);
}

// -------------------- NN lists (same as TSP) --------------------
void MFACO_CVRP::build_nn_lists() {
  nn_list.resize(n * k);
  backup_list.resize(n * bl);

  const int32_t total = k + bl;
  if (total <= 0)
    return;

  std::vector<Vec2d> pts(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const size_t off = static_cast<size_t>(i) * 2;
    pts[static_cast<size_t>(i)] =
        Vec2d{(double)coords[off], (double)coords[off + 1]};
  }

  KDTree shared_kdtree(pts, /*round_distances=*/false);

#pragma omp parallel default(none) shared(shared_kdtree) firstprivate(total)
  {
    KDTree kdtree = shared_kdtree;
    std::vector<int32_t> deleted_nodes;
    deleted_nodes.reserve(total + 8);

#pragma omp for schedule(static)
    for (int32_t u = 0; u < n; ++u) {
      deleted_nodes.clear();
      int32_t current_k = 0;
      int32_t current_bl = 0;

      // Force depot as first neighbor for customers
      if (u > 0 && k > 0) {
        nn_list[u * k + 0] = 0;
        current_k = 1;
      }

      // Search KDTree
      // We search a bit more than total to account for self/depot skips
      int32_t search_limit = total + 5;

      for (int32_t step = 0; step < search_limit; ++step) {
        if (current_k >= k && current_bl >= bl)
          break;

        uint32_t pt_idx = kdtree.nn_bottom_up(static_cast<uint32_t>(u));
        int32_t v = static_cast<int32_t>(pt_idx);

        kdtree.delete_point(pt_idx);
        deleted_nodes.push_back(v);

        if (v == u)
          continue; // Skip self
        if (u > 0 && v == 0)
          continue; // Skip depot for customers (already added)

        if (current_k < k) {
          nn_list[u * k + current_k] = v;
          current_k++;
        } else if (current_bl < bl) {
          backup_list[u * bl + current_bl] = v;
          current_bl++;
        }
      }

      // Undelete all deleted nodes
      for (int32_t v_del : deleted_nodes) {
        kdtree.undelete_point(static_cast<uint32_t>(v_del));
      }
    }
  }
}

// void MFACO_CVRP::build_nn_pos() { ... } REMOVED

void MFACO_CVRP::build_heuristic() {
  heuristic_sparse.resize(n * k);
  if (disable_heuristic) {
    std::fill(heuristic_sparse.begin(), heuristic_sparse.end(), 1.0f);
    return;
  }
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float d = dist(u, v);
      float d0 = dist(u, 0);
      float d1 = dist(0, v);
      // Savings heuristic
      d = d0 + d1 - d;
      heuristic_sparse[u * k + j] = d;
    }
  }
}

void MFACO_CVRP::build_d0() {
  d0.resize(n);
  for (int32_t v = 0; v < n; ++v) {
    d0[v] = dist(0, v);
  }
}

// -------------------- initial perm (greedy NN on customers, score by split)
// --------------------
void MFACO_CVRP::build_initial_solution() {
  float best_c = std::numeric_limits<float>::max();
  best_route.clear();

  int32_t num_starts = std::min(8, n);
  std::vector<int32_t> current_route;
  current_route.reserve(n * 2);
  std::vector<uint8_t> visited(n, 0);

  auto candidate_fits = [&](int32_t prev, int32_t v, int64_t load,
                            float route_time) -> bool {
    if (v <= 0 || v >= n || visited[v])
      return false;
    if (load + demand_int[v] > capacity_int)
      return false;
    if (!has_time_windows)
      return true;
    return can_append_tw(prev, v, route_time);
  };

  for (int32_t s = 0; s < num_starts; ++s) {
    current_route.clear();
    current_route.push_back(0);
    std::fill(visited.begin(), visited.end(), 0);
    visited[0] = 1;

    int32_t curr = 0;
    int64_t route_load = 0;
    float route_time = 0.0f;
    float cost = 0.0f;
    int32_t visited_cnt = 0;

    while (visited_cnt < m) {
      int32_t best_nb = -1;
      float min_d = std::numeric_limits<float>::max();

      int32_t lookup = curr;
      if (lookup >= 0 && lookup < n) {
        for (int32_t j = 0; j < k; ++j) {
          int32_t v = nn_list[lookup * k + j];
          if (candidate_fits(curr, v, route_load, route_time)) {
            best_nb = v;
            min_d = dist(curr, v);
            break;
          }
        }
      }

      if (best_nb == -1) {
        int32_t offset = (visited_cnt == 0 && n > 1) ? (s % (n - 1)) : 0;
        for (int32_t t = 0; t < m; ++t) {
          int32_t v = 1 + ((offset + t) % m);
          if (!candidate_fits(curr, v, route_load, route_time))
            continue;
          float d = dist(curr, v);
          if (d < min_d) {
            min_d = d;
            best_nb = v;
          }
        }
      }

      if (best_nb == -1) {
        if (curr == 0)
          break;
        cost += dist(curr, 0);
        current_route.push_back(0);
        curr = 0;
        route_load = 0;
        route_time = 0.0f;
        continue;
      }

      float arrival = route_time + dist(curr, best_nb);
      cost += dist(curr, best_nb);
      current_route.push_back(best_nb);
      visited[best_nb] = 1;
      visited_cnt++;
      route_load += demand_int[best_nb];
      route_time = has_time_windows ? std::max(arrival, ready_time[best_nb])
                                    : arrival;
      curr = best_nb;
    }

    if (curr != 0) {
      cost += dist(curr, 0);
      current_route.push_back(0);
    }

    if (visited_cnt == m && route_fully_feasible(current_route) && cost < best_c) {
      best_c = cost;
      best_route = current_route;
    }
  }

  if (best_route.empty()) {
    best_route.clear();
    best_route.push_back(0);
    for (int32_t v = 1; v < n; ++v) {
      best_route.push_back(v);
      best_route.push_back(0);
    }
    best_c = 0.0f;
    for (size_t i = 0; i + 1 < best_route.size(); ++i)
      best_c += dist(best_route[i], best_route[i + 1]);
  }

  source_cost = best_c;
  best_cost = best_c;
  source_route = best_route;

  tau_max = 1.0f / (rho * best_cost + EPS);
  tau_min = tau_max * 0.001f;
  std::fill(pheromone_sparse.begin(), pheromone_sparse.end(), tau_max);
}

// REMOVED split_dp, split_cost_fast, greedy_cost, decode_perm_to_route0

// -------------------- pheromone bounds (same formula as TSP)
// --------------------
std::pair<float, float>
MFACO_CVRP::calc_trail_limits_cl(float solution_cost) const {
  float tau_max_ = 1.0f / (solution_cost * (1.0f - rho) + EPS);
  float avg = static_cast<float>(std::max(2, k));
  float p = std::pow(p_best, 1.0f / avg);
  float tau_min_ =
      std::min(tau_max_, tau_max_ * (1.0f - p) / ((avg - 1.0f) * p + EPS));
  return {tau_min_, tau_max_};
}

std::pair<float, float>
MFACO_CVRP::calc_trail_limits_smooth(float solution_cost) const {
  (void)solution_cost;
  float tau_max_ = 1.0f;
  float denom = static_cast<float>(std::max<int32_t>(1, k));
  float tau_min_ = 1.0f / denom;
  return {tau_min_, tau_max_};
}

// -------------------- probmat (same as TSP) --------------------
void MFACO_CVRP::compute_probmat(const float *prior_ptr,
                                 std::vector<float> &probmat) {
  probmat.resize((size_t)n * (size_t)k);

  const float beta = 1.0f;  // if you want classic eta^beta
  const float gamma = 1.0f; // strength of learned prior
  const float eps = EPS;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    // ---- compute prior normalization stats for this row (u) ----
    float mean_z = 0.0f;
    float var_z = 0.0f;

    // if (prior_ptr) {
    //   // mean
    //   for (int32_t j = 0; j < k; ++j) {
    //     float z = prior_ptr[u * k + j];
    //     // optional clamp for safety (avoid huge exp)
    //     z = std::max(-10.0f, std::min(10.0f, z));
    //     mean_z += z;
    //   }
    //   mean_z /= (float)k;

    //   // variance
    //   for (int32_t j = 0; j < k; ++j) {
    //     float z = prior_ptr[u * k + j];
    //     z = std::max(-10.0f, std::min(10.0f, z));
    //     float dz = z - mean_z;
    //     var_z += dz * dz;
    //   }
    //   var_z /= (float)k;
    // }
    // float std_z = (prior_ptr ? std::sqrt(var_z + 1e-6f) : 1.0f);

    // ---- first pass: compute logits and max for stable exp ----
    float max_logit = -std::numeric_limits<float>::infinity();

    // store logits temporarily (stack or reuse probmat as scratch)
    // since k is small (32), a small local array is fine:
    float logits[MAX_CAND_LIST_SIZE];

    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;

      float tau = pheromone_sparse[idx];
      float eta = heuristic_sparse[idx];

      // Base: alpha*log(tau)
      float logit = alpha * std::log(tau + eps);

      // Heuristic: beta*log(eta)  (if disabled, eta==1 => log=0)
      if (!disable_heuristic) {
        logit += beta * std::log(eta + eps);
      }

      // Prior logits: gamma * normalized_z
      if (prior_ptr) {
        float z = prior_ptr[idx];
        // z = std::max(-10.0f, std::min(10.0f, z));
        // float z_norm = (z - mean_z) / std_z; // row-center + row-scale
        logit += gamma * z;
      }

      logits[j] = logit;
      if (logit > max_logit)
        max_logit = logit;
    }

    // ---- second pass: exp(logit - max) -> positive weights ----
    for (int32_t j = 0; j < k; ++j) {
      int32_t idx = u * k + j;
      float w = std::exp(logits[j] - max_logit);
      probmat[idx] = std::max(w, eps);
    }
  }
}

void MFACO_CVRP::sample(bool require_prob, const float *prior_ptr,
                        SampleResult &result, bool parallel_traced) {
  auto route_cost_euclid = [&](const std::vector<int32_t> &route) -> float {
    float c = 0.0f;
    if (route.size() < 2)
      return c;
    for (size_t i = 0; i + 1 < route.size(); ++i)
      c += dist(route[i], route[i + 1]);
    return c;
  };

  result.clear();
  result.costs.resize(n_ants);
  // CVRP solutions are represented as full depot-separated routes with
  // multiple 0s (e.g., 0 ... 0 ... 0).
  result.routes.resize(n_ants);
  result.decoded_routes.resize(n_ants);

  if (require_prob) {
    result.costs_raw.resize(n_ants);
    result.routes_raw.resize(n_ants);
    result.logps.resize(n_ants);
  }

  result.new_edges_count.resize(n_ants);
  result.edge_survival.resize(n_ants);

  std::vector<float> probmat;
  compute_probmat(prior_ptr, probmat);

  std::vector<int32_t> start_nodes(n_ants);
  for (int32_t a = 0; a < n_ants; ++a) {
    start_nodes[a] = 1 + (int32_t)rng_.next_uint((uint32_t)m);
  }

  std::vector<uint64_t> ant_seeds;
  auto ensure_ant_seeds = [&]() {
    if (!ant_seeds.empty())
      return;
    ant_seeds.resize((size_t)n_ants);
    for (int32_t a = 0; a < n_ants; ++a) {
      uint64_t hi = (uint64_t)rng_.next_u32();
      uint64_t lo = (uint64_t)rng_.next_u32();
      ant_seeds[(size_t)a] =
          (hi << 32) ^ lo ^ (0x9e3779b97f4a7c15ULL + (uint64_t)a);
    }
  };

  if (require_prob) {
    if (!parallel_traced) {
      result.traces.reserve(n_ants, n_ants * min_new_edges * 2);
      result.traces.starts.push_back(0);

      std::vector<int32_t> checklist;
      checklist.reserve(m);

      for (int32_t a = 0; a < n_ants; ++a) {
        // Trace construction into a decoded CVRP route (with depot zeros).
        result.decoded_routes[a].clear();
        std::vector<int32_t> route_raw_unused;

        MFACOTrace trace;
        trace.reserve(min_new_edges * 2);

        float logp_sum = 0.0f;
        int32_t mne_out = 0;

        float surv_out = 0.0f;
        (void)sample_ant_direct_traced(
            probmat.data(), start_nodes[a], result.decoded_routes[a],
            route_raw_unused, result.costs_raw[a], mne_out, checklist, trace,
            rng_, logp_sum, surv_out, prior_ptr);
        result.new_edges_count[a] = mne_out;
        result.edge_survival[a] = surv_out;
        result.logps[a] = logp_sum;

        // Canonicalize decoded route (some variants output a permutation only).
        if (result.decoded_routes[a].empty() ||
            result.decoded_routes[a].front() != 0)
          result.decoded_routes[a].insert(result.decoded_routes[a].begin(), 0);
        if (result.decoded_routes[a].back() != 0)
          result.decoded_routes[a].push_back(0);

        // Store the full depot-separated route and compute true CVRP cost.
        result.routes[a] = result.decoded_routes[a];
        result.costs[a] = route_cost_euclid(result.routes[a]);

        result.traces.start_nodes.push_back(trace.start_node);
        for (size_t i = 0; i < trace.curr_nodes.size(); ++i) {
          result.traces.curr_nodes.push_back(trace.curr_nodes[i]);
          result.traces.chosen_nodes.push_back(trace.chosen_nodes[i]);
          result.traces.is_stochastic.push_back(trace.is_stochastic[i]);
          result.traces.pick_j.push_back(trace.pick_j[i]);
          result.traces.valid_mask.push_back(trace.valid_mask[i]);
          result.traces.is_new_edge.push_back(trace.is_new_edge[i]);
        }
        result.traces.starts.push_back(
            (int32_t)result.traces.curr_nodes.size());
      }
    } else {
      ensure_ant_seeds();
      std::vector<MFACOTrace> traces_per_ant((size_t)n_ants);

#pragma omp parallel
      {
        std::vector<int32_t> checklist;
        checklist.reserve(m);

#pragma omp for schedule(static, 1)
        for (int32_t a = 0; a < n_ants; ++a) {
          MFACOTrace &trace = traces_per_ant[(size_t)a];
          trace.reserve(min_new_edges * 2);
          Xoshiro128Plus rng_local;
          rng_local.seed(ant_seeds[(size_t)a]);

          float logp_sum = 0.0f;
          int32_t mne_out = 0;

          float surv_out = 0.0f;
          (void)sample_ant_direct_traced(
              probmat.data(), start_nodes[a], result.routes[a],
              result.routes_raw[a], result.costs_raw[a], mne_out, checklist,
              trace, rng_local, logp_sum, surv_out, prior_ptr);
          result.new_edges_count[a] = mne_out;
          result.new_edges_count[a] = mne_out;
          result.edge_survival[a] = surv_out;
          result.logps[a] = logp_sum;

          // Canonicalize route format to depot-separated representation.
          if (result.routes[a].empty() || result.routes[a].front() != 0)
            result.routes[a].insert(result.routes[a].begin(), 0);
          if (result.routes[a].back() != 0)
            result.routes[a].push_back(0);

          if (!result.routes_raw[a].empty()) {
            if (result.routes_raw[a].front() != 0)
              result.routes_raw[a].insert(result.routes_raw[a].begin(), 0);
            if (result.routes_raw[a].back() != 0)
              result.routes_raw[a].push_back(0);
          }

          // Keep decoded_routes in sync for legacy return_decoded.
          result.decoded_routes[a] = result.routes[a];
          result.costs[a] = route_cost_euclid(result.routes[a]);
        }
      }

      // Merge traces in ant index order
      result.traces.clear();
      result.traces.starts.resize((size_t)n_ants + 1);
      result.traces.start_nodes.resize((size_t)n_ants);
      result.traces.starts[0] = 0;
      for (int32_t a = 0; a < n_ants; ++a) {
        const MFACOTrace &t = traces_per_ant[(size_t)a];
        result.traces.start_nodes[(size_t)a] = t.start_node;
        result.traces.starts[(size_t)a + 1] =
            result.traces.starts[(size_t)a] + (int32_t)t.curr_nodes.size();
      }
      int32_t total = result.traces.starts[(size_t)n_ants];
      result.traces.curr_nodes.resize((size_t)total);
      result.traces.chosen_nodes.resize((size_t)total);
      result.traces.is_stochastic.resize((size_t)total);
      result.traces.pick_j.resize((size_t)total);
      result.traces.valid_mask.resize((size_t)total);
      result.traces.is_new_edge.resize((size_t)total);

      for (int32_t a = 0; a < n_ants; ++a) {
        const MFACOTrace &t = traces_per_ant[(size_t)a];
        int32_t off = result.traces.starts[(size_t)a];
        for (size_t i = 0; i < t.curr_nodes.size(); ++i) {
          result.traces.curr_nodes[(size_t)off + i] = t.curr_nodes[i];
          result.traces.chosen_nodes[(size_t)off + i] = t.chosen_nodes[i];
          result.traces.is_stochastic[(size_t)off + i] = t.is_stochastic[i];
          result.traces.pick_j[(size_t)off + i] = t.pick_j[i];
          result.traces.valid_mask[(size_t)off + i] = t.valid_mask[i];
          result.traces.is_new_edge[(size_t)off + i] = t.is_new_edge[i];
        }
      }
    }
  } else {
    // Fast mode: parallel
    ensure_ant_seeds();
#pragma omp parallel
    {
      std::vector<int32_t> checklist;
      checklist.reserve(m);

#pragma omp for schedule(static, 1)
      for (int32_t a = 0; a < n_ants; ++a) {
        // Build a candidate solution; we will canonicalize to a full CVRP route
        // with depot (0) separators and compute the true CVRP cost.
        Xoshiro128Plus rng_local;
        rng_local.seed(ant_seeds[(size_t)a]);

        // Write whatever the sampler returns into routes[a] first. Some
        // sampler variants return only a customer permutation (no depot zeros).
        // We handle both and ensure we end with a depot-separated route.
        (void)sample_ant_direct(probmat.data(), start_nodes[a],
                                result.routes[a], result.new_edges_count[a],
                                checklist, rng_local, prior_ptr);

        // Ensure canonical start/end depot.
        if (result.routes[a].empty() || result.routes[a].front() != 0)
          result.routes[a].insert(result.routes[a].begin(), 0);
        if (result.routes[a].back() != 0)
          result.routes[a].push_back(0);

        // Keep decoded_routes in sync for legacy return_decoded.
        result.decoded_routes[a] = result.routes[a];

        // Recompute true CVRP cost over the depot-separated route.
        result.costs[a] = route_cost_euclid(result.routes[a]);
      }
    }
  }
}

// -------------------- update pheromone: deposit on decoded VRP edges
// --------------------
void MFACO_CVRP::update_pheromone(const std::vector<int32_t> &best_route_in,
                                  float new_best_cost) {
  if (has_time_windows && !route_fully_feasible(best_route_in)) {
    return;
  }

  // Update global best
  if (new_best_cost < best_cost) {
    best_cost = new_best_cost;
    best_route = best_route_in;
  }

  // Update trail limits based on global best
  auto [tmin, tmax] = smooth_mmas ? calc_trail_limits_smooth(best_cost)
                                  : calc_trail_limits_cl(best_cost);
  tau_min = tmin;
  tau_max = tmax;

  // Precompute positions for O(1) in-route check
  // For CVRP, best_route_in has depot 0 multiple times.
  // We map each customer node (1..n-1) to its index in best_route_in.
  int32_t R = (int32_t)best_route_in.size();
  std::vector<int32_t> pos(n, -1);
  for (int32_t i = 0; i < R; ++i) {
    int32_t v = best_route_in[i];
    if (v != 0) {
      pos[v] = i;
    }
  }

  auto in_route_edge = [&](int32_t u, int32_t v) -> bool {
    // Edge (u, v). If both are customers, they apply adjacency in best_route_in
    if (u != 0 && v != 0) {
      int32_t pu = pos[u];
      int32_t pv = pos[v];
      return std::abs(pu - pv) == 1;
    }

    // One is depot 0.
    if (u == 0 && v == 0)
      return false; // Loop (0,0) not relevant

    int32_t cust = (u != 0) ? u : v;
    int32_t p = pos[cust];
    // Check neighbors of cust in best_route_in
    if (p > 0 && best_route_in[p - 1] == 0)
      return true;
    if (p < R - 1 && best_route_in[p + 1] == 0)
      return true;

    return false;
  };

  const float decay_factor = 1.0f - rho;
  const float deposit = (!smooth_mmas) ? (1.0f / (new_best_cost + EPS)) : 0.0f;

#pragma omp parallel for schedule(static)
  for (int32_t u = 0; u < n; ++u) {
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      float &tau = pheromone_sparse[u * k + j];
      bool is_in = in_route_edge(u, v);

      if (smooth_mmas) {
        float target = is_in ? tau_max : tau_min;
        tau = decay_factor * tau + rho * target;
      } else {
        tau *= decay_factor;
        if (is_in) {
          tau += deposit;
        }
        tau = std::max(tau_min, std::min(tau_max, tau));
      }
    }
  }

  // Update source solution
  source_route = best_route_in;
  source_cost = new_best_cost;
}

// REMOVED two_opt_nn_prior

// -------------------- Inter-route LS --------------------

std::vector<std::vector<int32_t>> MFACO_CVRP::initial_routes_from_perm(
    const std::vector<int32_t> &solution) const {
  // Parse full route [0, c1..c2, 0, c3..c4, 0] into vector of routes
  // Each route should be [0, c1..c2, 0] (as expected by LS helpers)
  std::vector<std::vector<int32_t>> routes;

  // Solution should start with 0? sample_ant_direct output starts with customer
  // usually if called with start_node, but build_initial_solution adds 0s.
  // Parsing the full route into sub-routes.

  std::vector<int32_t> current;
  bool in_route = false;

  // Handling standard [0, ..., 0] form
  for (int32_t node : solution) {
    if (node == 0) {
      if (in_route) {
        current.push_back(0); // Close route
        routes.push_back(current);
        current.clear();
        in_route = false;
      }
      // Start new route potentially?
      // if next is customer, yes.
    } else {
      if (!in_route) {
        current.push_back(0); // Start new route
        in_route = true;
      }
      current.push_back(node);
    }
  }
  if (!current
           .empty()) { // Should imply last node wasn't 0 or route wasn't closed
    current.push_back(0);
    routes.push_back(current);
  }
  return routes;
}

void MFACO_CVRP::routes_to_perm(const std::vector<std::vector<int32_t>> &routes,
                                std::vector<int32_t> &solution,
                                std::vector<int32_t> &positions) {
  solution.clear();
  solution.reserve(n * 2); // Roughly
  positions.assign(
      n, -1); // Not really useful for full route? but might be used by LS?

  // Flatten routes: [0, r1, 0, r2, 0 ...]
  // Routes are [0, c.., 0]
  // We want to merge them. R1=[0, A, 0], R2=[0, B, 0] -> [0, A, 0, B, 0]

  bool first = true;
  int32_t idx = 0;

  for (const auto &r : routes) {
    if (r.size() < 2)
      continue; // Invalid

    // If not first, skip the leading 0 (it overlaps with previous trailing 0)?
    // OR strictly concat: [0, A, 0, 0, B, 0].
    // CVRP routes usually merge: [0, A, 0, B, 0].

    size_t start = 0;
    if (!first) {
      if (r[0] == 0)
        start = 1; // Skip leading 0
    }

    for (size_t i = start; i < r.size(); ++i) {
      int32_t node = r[i];
      solution.push_back(node);
      // Position map for customers
      if (node > 0) {
        // positions[node] = ...? In flattened array?
        // positions vector mainly used for Permutation logic.
      }
    }
    first = false;
  }
  // Ensure starts with 0? Yes if first route started with 0.
}

// ============================================================================
// Intra-Route Local Search (2-opt)
// ============================================================================

float MFACO_CVRP::intra_route_ls(std::vector<int32_t> &route,
                                 std::vector<int32_t> &checklist) {
  float total_improvement = 0.0f;
  // Build route structure: identify which route each node belongs to
  std::vector<int32_t> node_route(n, -1);
  std::vector<std::vector<int32_t>> routes;
  std::vector<int32_t> current;

  for (size_t i = 0; i < route.size(); ++i) {
    int32_t node = route[i];
    if (node == 0) {
      if (!current.empty()) {
        int32_t route_idx = (int32_t)routes.size();
        for (int32_t c : current) {
          node_route[c] = route_idx;
        }
        routes.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(node);
    }
  }

  if (routes.empty())
    return 0.0f;

  auto sequence_feasible = [&](const std::vector<int32_t> &seq) -> bool {
    int64_t route_load = 0;
    float route_time = 0.0f;
    int32_t prev = 0;

    for (int32_t node : seq) {
      if (node <= 0 || node >= n)
        return false;
      route_load += demand_int[node];
      if (route_load > capacity_int)
        return false;
      if (has_time_windows) {
        float arrival = route_time + dist(prev, node);
        if (arrival > due_time[node] + 1e-6f)
          return false;
        route_time = std::max(arrival, ready_time[node]);
        if (route_time + dist(node, 0) > due_time[0] + 1e-6f)
          return false;
      }
      prev = node;
    }

    if (has_time_windows)
      return route_time + dist(prev, 0) <= due_time[0] + 1e-6f;
    return true;
  };

  // Build positions within each route
  std::vector<int32_t> pos_in_route(n, -1);
  for (size_t r = 0; r < routes.size(); ++r) {
    for (size_t i = 0; i < routes[r].size(); ++i) {
      pos_in_route[routes[r][i]] = (int32_t)i;
    }
  }

  // Track nodes already in checklist to avoid duplicates
  std::vector<uint8_t> in_checklist_local(n, 0);
  for (int32_t node : checklist) {
    if (node > 0 && node < n) {
      in_checklist_local[node] = 1;
    }
  }

  // Process checklist - focused 2-opt
  size_t checklist_pos = 0;
  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a <= 0 || a >= n)
      continue;

    int32_t r = node_route[a];
    if (r < 0 || r >= (int32_t)routes.size())
      continue;

    auto &seq = routes[r];
    int32_t size = (int32_t)seq.size();
    if (size < 2)
      continue;

    int32_t a_pos = pos_in_route[a];
    if (a_pos < 0)
      continue;

    // Get neighbors of a in the route (including depot endpoints)
    int32_t a_prev = (a_pos > 0) ? seq[a_pos - 1] : 0;
    int32_t a_next = (a_pos < size - 1) ? seq[a_pos + 1] : 0;

    float dist_a_prev = dist(a_prev, a);
    float dist_a_next = dist(a, a_next);

    float max_diff = 0.0f;
    int32_t best_i = -1, best_j = -1;

    // Check 2-opt moves involving edge (a_prev, a)
    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue; // Must be same route

      float dist_ab = dist(a, b);
      if (dist_a_prev <= dist_ab)
        break; // NN list is sorted

      int32_t b_pos = pos_in_route[b];
      if (b_pos < 0 || b_pos == a_pos)
        continue;

      // Get b's predecessor
      int32_t b_prev = (b_pos > 0) ? seq[b_pos - 1] : 0;

      // 2-opt: reverse segment between a and b
      // Current: ...-a_prev-a-...-b_prev-b-...
      // New:     ...-a_prev-b_prev-...-a-b-...
      if (b_pos > a_pos) {
        // Reverse [a, b_prev]
        float d_old = dist_a_prev + dist(b_prev, b);
        float d_new = dist(a_prev, b_prev) + dist_ab;
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = a_pos;
          best_j = b_pos;
        }
      } else {
        // Reverse [b, a_prev]
        float d_old = dist(b_prev, b) + dist_a_prev;
        float d_new = dist(b_prev, a) + dist(b, a_prev);
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = b_pos;
          best_j = a_pos;
        }
      }
    }

    // Check 2-opt moves involving edge (a, a_next)
    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue;

      float dist_ab = dist(a, b);
      if (dist_a_next <= dist_ab)
        break;

      int32_t b_pos = pos_in_route[b];
      if (b_pos < 0 || b_pos == a_pos)
        continue;

      int32_t b_next = (b_pos < size - 1) ? seq[b_pos + 1] : 0;

      if (b_pos > a_pos) {
        // Reverse [a_next, b]
        float d_old = dist_a_next + dist(b, b_next);
        float d_new = dist_ab + dist(a_next, b_next);
        float diff = d_old - d_new;
        if (diff > max_diff) {
          max_diff = diff;
          best_i = a_pos + 1;
          best_j = b_pos + 1;
        }
      }
    }

    // Apply best move if found
    if (max_diff > 1e-6f && best_i >= 0 && best_j > best_i) {
      std::reverse(seq.begin() + best_i, seq.begin() + best_j);
      if (!sequence_feasible(seq)) {
        std::reverse(seq.begin() + best_i, seq.begin() + best_j);
        continue;
      }
      total_improvement += max_diff;

      // Update positions
      for (int32_t i = best_i; i < best_j; ++i) {
        pos_in_route[seq[i]] = i;
      }

      // Add affected nodes to checklist if extend_ls
      if (extend_ls) {
        for (int32_t i = std::max(0, best_i - 1);
             i < std::min(size, best_j + 1); ++i) {
          int32_t node = seq[i];
          if (node > 0 && node < n && !in_checklist_local[node]) {
            checklist.push_back(node);
            in_checklist_local[node] = 1;
          }
        }
      }
    }
  }

  // Reconstruct route from modified segments
  std::vector<int32_t> new_route;
  new_route.reserve(route.size());
  for (const auto &seg : routes) {
    new_route.push_back(0);
    for (int32_t c : seg) {
      new_route.push_back(c);
    }
  }
  new_route.push_back(0);
  route = new_route;
  return total_improvement;
}

float MFACO_CVRP::intra_route_oropt(std::vector<int32_t> &route,
                                    std::vector<int32_t> &checklist,
                                    int32_t segment_len) {
  if (segment_len < 1 || segment_len > 3)
    return 0.0f;

  float total_improvement = 0.0f;
  std::vector<int32_t> node_route(n, -1);
  std::vector<std::vector<int32_t>> routes;
  std::vector<int32_t> current;

  for (size_t i = 0; i < route.size(); ++i) {
    int32_t node = route[i];
    if (node == 0) {
      if (!current.empty()) {
        int32_t route_idx = (int32_t)routes.size();
        for (int32_t c : current) {
          node_route[c] = route_idx;
        }
        routes.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(node);
    }
  }

  if (routes.empty())
    return 0.0f;

  auto sequence_feasible = [&](const std::vector<int32_t> &seq) -> bool {
    int64_t route_load = 0;
    float route_time = 0.0f;
    int32_t prev = 0;

    for (int32_t node : seq) {
      if (node <= 0 || node >= n)
        return false;
      route_load += demand_int[node];
      if (route_load > capacity_int)
        return false;
      if (has_time_windows) {
        float arrival = route_time + dist(prev, node);
        if (arrival > due_time[node] + 1e-6f)
          return false;
        route_time = std::max(arrival, ready_time[node]);
        if (route_time + dist(node, 0) > due_time[0] + 1e-6f)
          return false;
      }
      prev = node;
    }

    if (has_time_windows)
      return route_time + dist(prev, 0) <= due_time[0] + 1e-6f;
    return true;
  };

  std::vector<int32_t> pos_in_route(n, -1);
  for (size_t r = 0; r < routes.size(); ++r) {
    for (size_t i = 0; i < routes[r].size(); ++i) {
      pos_in_route[routes[r][i]] = (int32_t)i;
    }
  }

  std::vector<uint8_t> in_checklist_local(n, 0);
  for (int32_t node : checklist) {
    if (node > 0 && node < n) {
      in_checklist_local[node] = 1;
    }
  }

  size_t checklist_pos = 0;
  while (checklist_pos < checklist.size()) {
    int32_t a = checklist[checklist_pos++];
    if (a <= 0 || a >= n)
      continue;

    int32_t r = node_route[a];
    if (r < 0 || r >= (int32_t)routes.size())
      continue;

    auto &seq = routes[r];
    int32_t size = (int32_t)seq.size();
    if (size <= segment_len)
      continue;

    int32_t start_pos = pos_in_route[a];
    if (start_pos < 0 || start_pos + segment_len > size)
      continue;

    int32_t end_pos = start_pos + segment_len - 1;
    int32_t seg_first = seq[start_pos];
    int32_t seg_last = seq[end_pos];
    int32_t prev_seg = (start_pos > 0) ? seq[start_pos - 1] : 0;
    int32_t next_seg = (end_pos < size - 1) ? seq[end_pos + 1] : 0;

    float max_diff = 0.0f;
    int32_t best_insert_pos = -1;

    for (int32_t jj = 0; jj < k; ++jj) {
      int32_t b = nn_list[a * k + jj];
      if (b <= 0 || b >= n)
        continue;
      if (node_route[b] != r)
        continue;

      int32_t b_pos = pos_in_route[b];
      if (b_pos < 0)
        continue;
      if (b_pos >= start_pos - 1 && b_pos <= end_pos)
        continue;

      int32_t b_next = (b_pos < size - 1) ? seq[b_pos + 1] : 0;
      float d_old = dist(prev_seg, seg_first) + dist(seg_last, next_seg) +
                    dist(b, b_next);
      float d_new = dist(prev_seg, next_seg) + dist(b, seg_first) +
                    dist(seg_last, b_next);
      float diff = d_old - d_new;
      if (diff > max_diff) {
        max_diff = diff;
        best_insert_pos = b_pos;
      }
    }

    if (max_diff > 1e-6f && best_insert_pos >= 0) {
      std::vector<int32_t> original_seq = seq;
      std::vector<int32_t> segment(seq.begin() + start_pos,
                                   seq.begin() + start_pos + segment_len);
      seq.erase(seq.begin() + start_pos,
                seq.begin() + start_pos + segment_len);
      int32_t adjusted_insert_pos = best_insert_pos;
      if (best_insert_pos > start_pos)
        adjusted_insert_pos -= segment_len;
      seq.insert(seq.begin() + adjusted_insert_pos + 1, segment.begin(),
                 segment.end());

      if (!sequence_feasible(seq)) {
        seq.swap(original_seq);
        continue;
      }

      total_improvement += max_diff;

      for (int32_t i = 0; i < size; ++i) {
        pos_in_route[seq[i]] = i;
      }

      if (extend_ls) {
        int32_t changed_begin = std::min(start_pos, adjusted_insert_pos + 1);
        int32_t changed_end = std::max(start_pos + segment_len,
                                       adjusted_insert_pos + 1 + segment_len);
        changed_begin = std::max(0, changed_begin - 1);
        changed_end = std::min(size, changed_end + 1);
        for (int32_t i = changed_begin; i < changed_end; ++i) {
          int32_t node = seq[i];
          if (node > 0 && node < n && !in_checklist_local[node]) {
            checklist.push_back(node);
            in_checklist_local[node] = 1;
          }
        }
      }
    }
  }

  std::vector<int32_t> new_route;
  new_route.reserve(route.size());
  for (const auto &seg : routes) {
    new_route.push_back(0);
    for (int32_t c : seg) {
      new_route.push_back(c);
    }
  }
  new_route.push_back(0);
  route = new_route;
  return total_improvement;
}

// ============================================================================
// Optimized Inter-Route Local Search (Linked List + DLB + O(1) Delta)
// ============================================================================

float MFACO_CVRP::inter_route_ls_optimized(std::vector<int32_t> &perm,
                                           std::vector<int32_t> &positions,
                                           std::vector<int32_t> &checklist,
                                           std::vector<uint8_t> &in_checklist) {
  float total_improvement = 0.0f;
  // 1. Initialization (Thread-Local Vectors)
  std::vector<int32_t> next_node(2 * n);
  std::vector<int32_t> prev_node(2 * n);
  std::vector<int32_t> node_route(2 * n);
  std::vector<int64_t> cum_demand(2 * n);
  // We'll resize route_loads after finding num_routes
  std::vector<int64_t> route_loads;
  std::vector<bool> dlb(n, false);

  // Helpers
  auto dist = [&](int32_t u, int32_t v) {
    int32_t ru = (u >= n) ? 0 : u;
    int32_t rv = (v >= n) ? 0 : v;
    float dx = coords[ru * 2] - coords[rv * 2];
    float dy = coords[ru * 2 + 1] - coords[rv * 2 + 1];
    return std::sqrt(dx * dx + dy * dy);
  };

  auto touch = [&](int32_t u) {
    if (u < n) {
      dlb[u] = false;
      if (!in_checklist[u]) {
        checklist.push_back(u);
        in_checklist[u] = 1;
      }
    }
  };

  auto update_route_state = [&](int32_t start_node, int32_t route_id) {
    int32_t curr = start_node;
    if (curr < n)
      return;

    cum_demand[curr] = 0;
    node_route[curr] = route_id;
    int64_t load = 0;

    curr = next_node[curr];
    while (curr < n) {
      load += demand_int[curr];
      cum_demand[curr] = load;
      node_route[curr] = route_id;
      curr = next_node[curr];
    }
    route_loads[route_id] = load;
  };

  auto affected_routes_feasible = [&](int32_t r_a, int32_t r_b) {
    return linked_route_feasible(r_a, next_node, node_route, route_loads) &&
           linked_route_feasible(r_b, next_node, node_route, route_loads);
  };

  // Focused LS: only process nodes in checklist (no fallback to all nodes)
  if (checklist.empty()) {
    return 0.0f; // Nothing to do
  }

  // Get current routes
  auto routes =
      initial_routes_from_perm(perm); // Uses "perm" which is now full route
  int32_t num_routes = (int32_t)routes.size();

  if (num_routes > n) {
    // Just in case, though guaranteed by split logic usually
    next_node.resize(n + num_routes);
    prev_node.resize(n + num_routes);
    node_route.resize(n + num_routes);
    cum_demand.resize(n + num_routes);
  }
  route_loads.resize(num_routes);

  // Build Linked List from Routes
  for (int r = 0; r < num_routes; ++r) {
    const auto &current_route = routes[r];
    // current_route is [0, c1, ..., ck, 0]

    int32_t depot = n + r;
    int32_t prev = depot;
    int64_t load = 0;

    node_route[depot] = r;
    cum_demand[depot] = 0;

    // Iterate customers (skip first 0, last 0)
    for (size_t i = 1; i < current_route.size() - 1; ++i) {
      int32_t u = current_route[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      cum_demand[u] = load;
      prev = u;
    }
    // Close loop to depot
    next_node[prev] = depot;
    prev_node[depot] = prev;
    route_loads[r] = load;
  }

  const float EPS = 1e-5f; // Tighten EPS

  // 2. Main Loop
  int32_t head = 0;
  while (head < (int32_t)checklist.size()) {
    int32_t u = checklist[head++];
    in_checklist[u] = 0;

    if (dlb[u])
      continue;

    bool improved = false;
    int32_t r_u = node_route[u];

    // Check neighbors
    for (int32_t j = 0; j < k; ++j) {
      int32_t v = nn_list[u * k + j];
      if (v == 0)
        continue;

      int32_t r_v = node_route[v];

      // Pruning removed (was unsafe for Relocate/Swap involving prev edges)
      int32_t next_u = next_node[u];
      int32_t next_v = next_node[v];
      int32_t prev_v = prev_node[v];
      /*
      if (dist(u, v) > dist(u, next_u) + dist(v, next_v) + EPS) {
         continue;
      }
      */

      // A. Relocate u after v
      if (use_relocate && r_u != r_v) {
        if (route_loads[r_v] + demand_int[u] <= capacity_int) {
          // Case 1: Insert After v
          int32_t prev_u = prev_node[u];
          float delta = dist(prev_u, next_u) + dist(v, u) + dist(u, next_v) -
                        dist(prev_u, u) - dist(u, next_u) - dist(v, next_v);

          if (delta < -EPS) {
            // Unlink u
            next_node[prev_u] = next_u;
            prev_node[next_u] = prev_u;
            // Link u after v
            int32_t old_next_v = next_node[v];
            next_node[v] = u;
            prev_node[u] = v;
            next_node[u] = old_next_v;
            prev_node[old_next_v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!linked_route_feasible(r_u, next_node, node_route,
                                       route_loads) ||
                !linked_route_feasible(r_v, next_node, node_route,
                                       route_loads)) {
              next_node[prev_u] = u;
              prev_node[u] = prev_u;
              next_node[u] = next_u;
              prev_node[next_u] = u;
              next_node[v] = old_next_v;
              prev_node[old_next_v] = v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }
            touch(u);
            touch(v);
            touch(prev_u);
            touch(next_u);
            touch(old_next_v);
            improved = true;
            total_improvement -= delta; // delta is negative
            break;
          }

          // Case 2: Insert Before v (After prev_v)
          // Effectively: Relocate u after prev_v
          // Only possible if we didn't do Case 1 (improved=false)
          // But r_v is same. prev_v might be depot.
          // Check if prev_v is actually valid target (it is, since r_prev_v ==
          // r_v != r_u)

          float delta2 = dist(prev_u, next_u) + dist(prev_v, u) + dist(u, v) -
                         dist(prev_u, u) - dist(u, next_u) - dist(prev_v, v);

          if (delta2 < -EPS) {
            // Unlink u
            next_node[prev_u] = next_u;
            prev_node[next_u] = prev_u;
            // Link u after prev_v
            // current next of prev_v is v.
            next_node[prev_v] = u;
            prev_node[u] = prev_v;
            next_node[u] = v;
            prev_node[v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!linked_route_feasible(r_u, next_node, node_route,
                                       route_loads) ||
                !linked_route_feasible(r_v, next_node, node_route,
                                       route_loads)) {
              next_node[prev_u] = u;
              prev_node[u] = prev_u;
              next_node[u] = next_u;
              prev_node[next_u] = u;
              next_node[prev_v] = v;
              prev_node[v] = prev_v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }
            // Touches
            touch(u);
            touch(prev_v);
            touch(v);
            touch(prev_u);
            touch(next_u);
            improved = true;
            total_improvement -= delta2;
            break;
          }
        }
      }

      // B. Or-opt(2,0): relocate consecutive pair u,next_u after v.
      if (r_u != r_v && next_u < n) {
        int32_t u2 = next_u;
        int32_t next_u2 = next_node[u2];
        int64_t segment_load = demand_int[u] + demand_int[u2];
        if (route_loads[r_v] + segment_load <= capacity_int) {
          int32_t prev_u = prev_node[u];
          float delta = dist(prev_u, next_u2) + dist(v, u) +
                        dist(u2, next_v) - dist(prev_u, u) -
                        dist(u2, next_u2) - dist(v, next_v);

          if (delta < -EPS) {
            next_node[prev_u] = next_u2;
            prev_node[next_u2] = prev_u;
            next_node[v] = u;
            prev_node[u] = v;
            next_node[u2] = next_v;
            prev_node[next_v] = u2;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!affected_routes_feasible(r_u, r_v)) {
              next_node[prev_u] = u;
              prev_node[u] = prev_u;
              next_node[u2] = next_u2;
              prev_node[next_u2] = u2;
              next_node[v] = next_v;
              prev_node[next_v] = v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }

            touch(prev_u);
            touch(u);
            touch(u2);
            touch(next_u2);
            touch(v);
            touch(next_v);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }

      // C. Reversed Or-opt(2,0): relocate pair u,next_u as next_u,u.
      if (r_u != r_v && next_u < n) {
        int32_t u2 = next_u;
        int32_t next_u2 = next_node[u2];
        int64_t segment_load = demand_int[u] + demand_int[u2];
        if (route_loads[r_v] + segment_load <= capacity_int) {
          int32_t prev_u = prev_node[u];
          float delta = dist(prev_u, next_u2) + dist(v, u2) +
                        dist(u, next_v) - dist(prev_u, u) -
                        dist(u2, next_u2) - dist(v, next_v);

          if (delta < -EPS) {
            next_node[prev_u] = next_u2;
            prev_node[next_u2] = prev_u;
            next_node[v] = u2;
            prev_node[u2] = v;
            next_node[u2] = u;
            prev_node[u] = u2;
            next_node[u] = next_v;
            prev_node[next_v] = u;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!affected_routes_feasible(r_u, r_v)) {
              next_node[prev_u] = u;
              prev_node[u] = prev_u;
              next_node[u] = u2;
              prev_node[u2] = u;
              next_node[u2] = next_u2;
              prev_node[next_u2] = u2;
              next_node[v] = next_v;
              prev_node[next_v] = v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }

            touch(prev_u);
            touch(u);
            touch(u2);
            touch(next_u2);
            touch(v);
            touch(next_v);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }

      // D. Swap u, v
      if (use_swap && r_u != r_v) {
        int64_t load_u_new = route_loads[r_u] - demand_int[u] + demand_int[v];
        int64_t load_v_new = route_loads[r_v] - demand_int[v] + demand_int[u];

        if (load_u_new <= capacity_int && load_v_new <= capacity_int) {
          int32_t prev_u = prev_node[u];
          int32_t prev_v = prev_node[v];
          float delta = dist(prev_u, v) + dist(v, next_u) + dist(prev_v, u) +
                        dist(u, next_v) - dist(prev_u, u) - dist(u, next_u) -
                        dist(prev_v, v) - dist(v, next_v);
          if (delta < -EPS) {
            int32_t nu = next_node[u], pu = prev_node[u];
            int32_t nv = next_node[v], pv = prev_node[v];
            next_node[pu] = v;
            prev_node[nu] = v;
            next_node[pv] = u;
            prev_node[nv] = u;
            next_node[u] = nv;
            prev_node[u] = pv;
            next_node[v] = nu;
            prev_node[v] = pu;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!linked_route_feasible(r_u, next_node, node_route,
                                       route_loads) ||
                !linked_route_feasible(r_v, next_node, node_route,
                                       route_loads)) {
              next_node[pu] = u;
              prev_node[u] = pu;
              next_node[u] = nu;
              prev_node[nu] = u;
              next_node[pv] = v;
              prev_node[v] = pv;
              next_node[v] = nv;
              prev_node[nv] = v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }
            touch(u);
            touch(v);
            touch(pu);
            touch(nu);
            touch(pv);
            touch(nv);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }


      // E. 2-Opt*
      if (use_2opt_star && r_u != r_v) {
        int64_t head_u = cum_demand[u];
        int64_t tail_u = route_loads[r_u] - head_u;
        int64_t head_v = cum_demand[v];
        int64_t tail_v = route_loads[r_v] - head_v;

        if (head_u + tail_v <= capacity_int &&
            head_v + tail_u <= capacity_int) {
          float delta = dist(u, next_v) + dist(v, next_u) - dist(u, next_u) -
                        dist(v, next_v);
          if (delta < -EPS) {
            int32_t nu = next_node[u];
            int32_t nv = next_node[v];
            next_node[u] = nv;
            prev_node[nv] = u;
            next_node[v] = nu;
            prev_node[nu] = v;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!linked_route_feasible(r_u, next_node, node_route,
                                       route_loads) ||
                !linked_route_feasible(r_v, next_node, node_route,
                                       route_loads)) {
              next_node[u] = nu;
              prev_node[nu] = u;
              next_node[v] = nv;
              prev_node[nv] = v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }
            touch(u);
            touch(v);
            touch(nu);
            touch(nv);
            improved = true;
            total_improvement -= delta;
            break;
          }
      // F. Exchange(2,1): exchange consecutive pair u,next_u with v.
      if (r_u != r_v && next_u < n) {
        int32_t u2 = next_u;
        int32_t next_u2 = next_node[u2];
        int32_t prev_u = prev_node[u];
        int32_t prev_v = prev_node[v];
        int64_t segment_load = demand_int[u] + demand_int[u2];
        int64_t load_u_new = route_loads[r_u] - segment_load + demand_int[v];
        int64_t load_v_new = route_loads[r_v] - demand_int[v] + segment_load;

        if (load_u_new <= capacity_int && load_v_new <= capacity_int) {
          float delta = dist(prev_u, v) + dist(v, next_u2) + dist(prev_v, u) +
                        dist(u2, next_v) - dist(prev_u, u) -
                        dist(u2, next_u2) - dist(prev_v, v) -
                        dist(v, next_v);

          if (delta < -EPS) {
            next_node[prev_u] = v;
            prev_node[v] = prev_u;
            next_node[v] = next_u2;
            prev_node[next_u2] = v;
            next_node[prev_v] = u;
            prev_node[u] = prev_v;
            next_node[u2] = next_v;
            prev_node[next_v] = u2;

            update_route_state(n + r_u, r_u);
            update_route_state(n + r_v, r_v);
            if (!affected_routes_feasible(r_u, r_v)) {
              next_node[prev_u] = u;
              prev_node[u] = prev_u;
              next_node[u2] = next_u2;
              prev_node[next_u2] = u2;
              next_node[prev_v] = v;
              prev_node[v] = prev_v;
              next_node[v] = next_v;
              prev_node[next_v] = v;
              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              continue;
            }

            touch(prev_u);
            touch(u);
            touch(u2);
            touch(next_u2);
            touch(prev_v);
            touch(v);
            touch(next_v);
            improved = true;
            total_improvement -= delta;
            break;
          }
        }
      }

        }
      }
      // G. Or-opt(3,0): lower-priority candidate-restricted segment move.
      if (r_u != r_v && next_u < n) {
        int32_t u2 = next_u;
        int32_t u3 = next_node[u2];
        if (u3 < n) {
          int32_t next_u3 = next_node[u3];
          int64_t segment_load = demand_int[u] + demand_int[u2] + demand_int[u3];
          if (route_loads[r_v] + segment_load <= capacity_int) {
            int32_t prev_u = prev_node[u];
            float delta = dist(prev_u, next_u3) + dist(v, u) +
                          dist(u3, next_v) - dist(prev_u, u) -
                          dist(u3, next_u3) - dist(v, next_v);

            if (delta < -EPS) {
              next_node[prev_u] = next_u3;
              prev_node[next_u3] = prev_u;
              next_node[v] = u;
              prev_node[u] = v;
              next_node[u3] = next_v;
              prev_node[next_v] = u3;

              update_route_state(n + r_u, r_u);
              update_route_state(n + r_v, r_v);
              if (!affected_routes_feasible(r_u, r_v)) {
                next_node[prev_u] = u;
                prev_node[u] = prev_u;
                next_node[u3] = next_u3;
                prev_node[next_u3] = u3;
                next_node[v] = next_v;
                prev_node[next_v] = v;
                update_route_state(n + r_u, r_u);
                update_route_state(n + r_v, r_v);
                continue;
              }

              touch(prev_u);
              touch(u);
              touch(u2);
              touch(u3);
              touch(next_u3);
              touch(v);
              touch(next_v);
              improved = true;
              total_improvement -= delta;
              break;
            }
          }
        }
      }

    }
    if (improved)
      head = 0;
    else
      dlb[u] = true;
  }

  // 3. Reconstruct depot-separated route (not permutation)
  perm.clear();
  perm.reserve(m + num_routes + 1);
  for (int r = 0; r < num_routes; ++r) {
    int32_t depot_node = n + r;
    int32_t curr = next_node[depot_node];
    // Skip empty routes
    if (curr >= n)
      continue;
    perm.push_back(0); // Start depot
    while (curr < n) {
      perm.push_back(curr);
      curr = next_node[curr];
    }
  }
  if (!perm.empty() && perm.back() != 0)
    perm.push_back(0); // End depot

  return total_improvement;
}

// MFACO_CVRP::sample_ant_direct / traced (Relocation-Based Sampling)
// ============================================================================

float MFACO_CVRP::sample_ant_direct(const float *probmat, int32_t start_node,
                                    std::vector<int32_t> &route_out,
                                    int32_t &new_edges_out,
                                    std::vector<int32_t> &checklist,
                                    Xoshiro128Plus &rng, const float *prior) {
  // Relocation-Based Sampling (Phase 1) with Linked List & Split Logic

  // 1. Build Adjacency of Source (for new edge detection)
  std::vector<int32_t> src_prev(n, -1);
  std::vector<int32_t> src_next(n, -1);
  std::vector<uint8_t> src_adj_to_depot(n, 0);

  // Original route IDs for cross-route checks
  std::vector<int32_t> source_node_route_id(n, -1);
  int32_t current_route_id = 0;
  for (size_t i = 0; i < source_route.size(); ++i) {
    int32_t u = source_route[i];
    if (u == 0) {
      if (i > 0 && source_route[i - 1] != 0) {
        current_route_id++;
      }
    } else {
      source_node_route_id[u] = current_route_id;
    }
  }

  for (size_t i = 0; i + 1 < source_route.size(); ++i) {
    int32_t u = source_route[i];
    int32_t v = source_route[i + 1];
    if (u > 0 && u < n && v > 0 && v < n) {
      src_next[u] = v;
      src_prev[v] = u;
    } else {
      if (u == 0 && v > 0 && v < n)
        src_adj_to_depot[v] = 1;
      if (v == 0 && u > 0 && u < n)
        src_adj_to_depot[u] = 1;
    }
  }

  auto is_source_edge = [&](int32_t u, int32_t v) -> bool {
    if (u >= n || v >= n)
      return false;
    if (u == 0)
      return src_adj_to_depot[v];
    if (v == 0)
      return src_adj_to_depot[u];
    return (src_next[u] == v || src_prev[u] == v);
  };

  // 2. Initialize Linked List from Source Solution
  // Routes: [0, c1..ck, 0]. In LL: depot_node -> c1 -> ... -> ck -> depot_node
  std::vector<std::vector<int32_t>> init_routes =
      initial_routes_from_perm(source_route);
  int32_t num_routes = (int32_t)init_routes.size();
  int32_t max_routes = n; // Allow growth

  // Structures
  std::vector<int32_t> next_node(n + max_routes);
  std::vector<int32_t> prev_node(n + max_routes);
  std::vector<int32_t> node_route(n, -1); // Only for customers 1..n-1
  std::vector<int64_t> route_loads(max_routes, 0);

  for (int r = 0; r < num_routes; ++r) {
    int32_t depot = n + r;
    int32_t prev = depot;
    int64_t load = 0;

    const auto &rt = init_routes[r];
    // rt is [0, c1...ck, 0]
    for (size_t i = 1; i < rt.size() - 1; ++i) {
      int32_t u = rt[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      prev = u;
    }
    next_node[prev] = depot;
    prev_node[depot] = prev;
    route_loads[r] = load;
  }

  // 3. Setup Sampling
  std::vector<uint8_t> visited(n, 0);
  int32_t visited_count = 0;

  int32_t curr = start_node;
  if (curr <= 0 || curr >= n) {
    // Robust start node selection
    int attempts = 0;
    while (attempts < 10) {
      curr = 1 + (int32_t)rng.next_uint((uint32_t)m);
      if (curr > 0 && curr < n)
        break;
      attempts++;
    }
    if (curr <= 0 || curr >= n)
      curr = 1;
  }

  visited[curr] = 1;
  visited_count++;

  checklist.clear();
  checklist.push_back(curr);
  std::vector<uint8_t> in_checklist(n, 0);
  in_checklist[curr] = 1;

  int32_t new_edges_all = 0;
  int32_t new_edges_cross = 0;
  int32_t steps = 0;
  int32_t max_steps = m * 4;
  if (fixed_steps > 0)
    max_steps = fixed_steps;

  // 4. Main Relocation Loop
  while (true) {
    // Termination Check
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges_cross >= min_new_edges || visited_count >= m)
        break;
    }
    if (steps > max_steps)
      break;

    // Determine Current Route info
    int32_t r_curr;
    if (curr >= n) {
      r_curr = curr - n;
    } else {
      r_curr = node_route[curr];
    }

    // Select Next Node 'v'
    int16_t pick_j = -1;
    uint64_t valid_mask = 0;
    int32_t lookup_node = (curr >= n) ? 0 : curr;
    const float *row_prob = probmat + (size_t)lookup_node * k;
    auto [chosen, is_stoch, log_prob] =
        select_next_node(curr, r_curr, row_prob, visited, node_route,
                         route_loads, next_node, prev_node, num_routes,
                         max_routes, rng, pick_j, valid_mask);
    (void)is_stoch;
    (void)log_prob;
    (void)pick_j;
    (void)valid_mask;

    if (chosen == -1)
      break;

    // Execute Transition
    // 1. If v == 0: Split / End Route
    if (chosen == 0) {
      int32_t next_c = next_node[curr];

      // If curr is already at end of route (next is depot), just step.
      if (next_c >= n) {
        // Simply traverse to depot
        // Stats
        bool is_cross = true; // depot edge is cross/boundary
        if (is_cross)
          new_edges_cross++; // Count moving to depot as perturbation?
        if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
          new_edges_all++;
        }

        curr = next_c;
        steps++;
        continue;
      }

      // Else: Split.
      if (num_routes >= max_routes) {
        break;
      }

      // Insert new depot after curr
      int32_t r_new = num_routes++;
      int32_t new_depot = n + r_new;
      int32_t old_depot = n + r_curr;

      // Find end of old route (it connects to old_depot)
      int32_t route_end = prev_node[old_depot];

      // Relinking
      next_node[curr] = old_depot;
      prev_node[old_depot] = curr;

      next_node[new_depot] = next_c;
      prev_node[next_c] = new_depot;

      next_node[route_end] = new_depot;
      prev_node[new_depot] = route_end;

      // Update Loads & Route IDs for the new segment
      int64_t load_shift = 0;
      int32_t w = next_c;
      while (w != new_depot && w < n) {
        node_route[w] = r_new;
        load_shift += demand_int[w];
        w = next_node[w];
      }
      route_loads[r_curr] -= load_shift;
      route_loads[r_new] = load_shift;

      // Stats for split edge (curr, 0)
      if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
        new_edges_all++;
        if (!in_checklist[curr >= n ? 0 : curr]) {
          checklist.push_back(curr >= n ? 0 : curr);
          in_checklist[curr >= n ? 0 : curr] = 1;
        }
        new_edges_cross++;
      }

      curr = new_depot; // We are now at the start of new route
      steps++;
      continue;
    }

    // 2. If v is Customer: Relocate v after curr
    int32_t v = chosen;
    int32_t r_v = node_route[v];

    // Unlink v
    int32_t prev_v = prev_node[v];
    int32_t next_v = next_node[v];

    if (prev_v == curr) {
      // Already after curr? Just Traverse
      visited[v] = 1;
      visited_count++;
      curr = v;
      steps++;
      continue;
    }

    next_node[prev_v] = next_v;
    prev_node[next_v] = prev_v;

    // Insert v after curr
    int32_t next_c = next_node[curr];
    next_node[curr] = v;
    prev_node[v] = curr;
    next_node[v] = next_c;
    prev_node[next_c] = v;

    // Updates
    if (r_curr != r_v) {
      if (r_v != -1) {
        route_loads[r_v] -= demand_int[v];
      }
      route_loads[r_curr] += demand_int[v];
      node_route[v] = r_curr;
    }

    // Stats
    visited[v] = 1;
    visited_count++;

    int32_t u_idx = (curr >= n) ? 0 : curr;
    if (!is_source_edge(u_idx, v)) {
      new_edges_all++;
      if (!in_checklist[u_idx]) {
        checklist.push_back(u_idx);
        in_checklist[u_idx] = 1;
      }
      if (!in_checklist[v]) {
        checklist.push_back(v);
        in_checklist[v] = 1;
      }

      // Cross check
      bool is_cross = false;
      if (u_idx == 0)
        is_cross = true; // Depot->Node is boundary edge
      else if (source_node_route_id[u_idx] != source_node_route_id[v])
        is_cross = true;

      if (is_cross)
        new_edges_cross++;
    }

    curr = v;
    steps++;
  }

  new_edges_out = new_edges_cross;

  // 5. Flatten Routes
  route_out.clear();
  route_out.reserve(m + num_routes + 2);

  for (int r = 0; r < num_routes; ++r) {
    int32_t d = n + r;
    int32_t w = next_node[d];
    if (w >= n)
      continue; // Empty

    route_out.push_back(0);
    while (w < n) {
      route_out.push_back(w);
      w = next_node[w];
    }
  }
  if (!route_out.empty())
    route_out.push_back(0);

  enforce_time_windows(route_out);
  std::vector<int32_t> route_before_ls;
  if (has_time_windows)
    route_before_ls = route_out;

  // 6. Apply Local Search
  if (use_local_search && !checklist.empty()) {
    // Intra-Route Or-opt (1/2/3)
    intra_route_oropt(route_out, checklist, 1);
    intra_route_oropt(route_out, checklist, 2);
    intra_route_oropt(route_out, checklist, 3);
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
    // Inter-Route LS
    std::vector<int32_t> pos_ls(n, -1);
    inter_route_ls_optimized(route_out, pos_ls, checklist, in_checklist);
    // Intra-Route Or-opt (1/2/3)
    intra_route_oropt(route_out, checklist, 1);
    intra_route_oropt(route_out, checklist, 2);
    intra_route_oropt(route_out, checklist, 3);
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
  }

  if (has_time_windows && !route_fully_feasible(route_out))
    route_out.swap(route_before_ls);
  if (has_time_windows && !route_fully_feasible(route_out)) {
    enforce_time_windows(route_out);
    if (!route_fully_feasible(route_out))
      route_out = source_route;
  }

  // Calculate final cost
  float final_cost = 0.0f;
  for (size_t i = 0; i + 1 < route_out.size(); ++i) {
    final_cost += dist(route_out[i], route_out[i + 1]);
  }

  return final_cost;
}

std::tuple<int32_t, bool, float> MFACO_CVRP::select_next_node(
    int32_t curr, int32_t curr_route, const float *probmat_row,
    const std::vector<uint8_t> &visited, const std::vector<int32_t> &node_route,
    const std::vector<int64_t> &route_loads,
    const std::vector<int32_t> &next_node,
    const std::vector<int32_t> &prev_node, int32_t num_routes,
    int32_t max_routes, Xoshiro128Plus &rng, int16_t &out_pick_j,
    uint64_t &out_valid_mask) {
  int32_t c_size = 0;
  float sum_prob = 0.0f;
  out_valid_mask = 0;
  out_pick_j = -1;

  // Arrays on stack
  int32_t candidates[MAX_CAND_LIST_SIZE + 1];
  float probs[MAX_CAND_LIST_SIZE + 1];
  int16_t j_indices[MAX_CAND_LIST_SIZE + 1];

  int32_t lookup_node = (curr >= n) ? 0 : curr;

  auto route_state_feasible = [&](const std::vector<int32_t> &nn,
                                  const std::vector<int32_t> &nr,
                                  const std::vector<int64_t> &rl,
                                  int32_t routes) -> bool {
    if (!has_time_windows)
      return true;
    return linked_solution_feasible(routes, nn, nr, rl);
  };

  auto can_take_customer = [&](int32_t v) -> bool {
    if (v <= 0 || v >= n || visited[v])
      return false;
    int32_t v_route = node_route[v];
    if (curr_route < 0 || v_route < 0)
      return false;
    if (curr_route != v_route && route_loads[curr_route] + demand_int[v] > capacity_int)
      return false;
    if (!has_time_windows)
      return true;

    std::vector<int32_t> nn = next_node;
    std::vector<int32_t> pp = prev_node;
    std::vector<int32_t> nr = node_route;
    std::vector<int64_t> rl = route_loads;

    int32_t prev_v = pp[v];
    int32_t next_v = nn[v];
    if (prev_v == curr)
      return linked_route_feasible(curr_route, nn, nr, rl);

    nn[prev_v] = next_v;
    pp[next_v] = prev_v;

    int32_t next_c = nn[curr];
    nn[curr] = v;
    pp[v] = curr;
    nn[v] = next_c;
    pp[next_c] = v;

    if (curr_route != v_route) {
      rl[v_route] -= demand_int[v];
      rl[curr_route] += demand_int[v];
      nr[v] = curr_route;
    }

    return route_state_feasible(nn, nr, rl, num_routes);
  };

  auto can_take_depot = [&]() -> bool {
    if (curr >= n || curr_route < 0)
      return false;
    if (!has_time_windows)
      return true;
    int32_t next_c = next_node[curr];
    if (next_c >= n)
      return linked_route_feasible(curr_route, next_node, node_route, route_loads);
    if (num_routes >= max_routes)
      return false;

    std::vector<int32_t> nn = next_node;
    std::vector<int32_t> pp = prev_node;
    std::vector<int32_t> nr = node_route;
    std::vector<int64_t> rl = route_loads;

    int32_t r_new = num_routes;
    int32_t new_depot = n + r_new;
    int32_t old_depot = n + curr_route;
    int32_t route_end = pp[old_depot];

    nn[curr] = old_depot;
    pp[old_depot] = curr;
    nn[new_depot] = next_c;
    pp[next_c] = new_depot;
    nn[route_end] = new_depot;
    pp[new_depot] = route_end;

    int64_t load_shift = 0;
    int32_t w = next_c;
    int32_t guard = 0;
    while (w != new_depot && w < n) {
      nr[w] = r_new;
      load_shift += demand_int[w];
      w = nn[w];
      if (++guard > 2 * n)
        return false;
    }
    rl[curr_route] -= load_shift;
    rl[r_new] = load_shift;

    return route_state_feasible(nn, nr, rl, num_routes + 1);
  };

  // A. NN List
  for (int32_t j = 0; j < k; ++j) {
    int32_t v = nn_list[lookup_node * k + j];
    if (v < 0 || v >= n)
      continue;
    if (v == curr)
      continue;
    if (visited[v])
      continue;

    if (can_take_customer(v)) {
      float p = probmat_row[j];
      candidates[c_size] = v;
      probs[c_size] = p;
      j_indices[c_size] = (int16_t)j;
      if (j < 64)
        out_valid_mask |= (1ULL << j);
      sum_prob += p;
      c_size++;
    }
  }

  // B. Backup List
  if (c_size == 0) {
    for (int32_t j = 0; j < bl; ++j) {
      int32_t v = backup_list[lookup_node * bl + j];
      if (v < 0 || v >= n)
        continue;
      if (visited[v])
        continue;

      if (can_take_customer(v)) {
        candidates[c_size] = v;
        probs[c_size] = 1.0f;
        j_indices[c_size] = -1;
        sum_prob += 1.0f;
        c_size++;
        break;
      }
    }
  }

  // C. Global Fallback
  if (c_size == 0) {
    float min_d = std::numeric_limits<float>::max();
    int32_t best_global = -1;

    for (int32_t v = 1; v < n; ++v) {
      if (!visited[v]) {
        if (can_take_customer(v)) {
          float d = dist(lookup_node, v);
          if (d < min_d) {
            min_d = d;
            best_global = v;
          }
        }
      }

    }

    if (best_global == -1) {
      if (can_take_depot())
        best_global = 0;
    } else {
      float d_depot = dist(lookup_node, 0);
      if (d_depot < min_d && can_take_depot()) {
        best_global = 0;
      }
    }

    if (best_global != -1) {
      candidates[c_size] = best_global;
      probs[c_size] = 1.0f;
      j_indices[c_size] = -1;
      sum_prob += 1.0f;
      c_size++;
    }
  }

  if (c_size == 0) {
    return {-1, false, 0.0f};
  }

  // Selection
  bool is_stoch = (c_size > 1);
  int32_t chosen = candidates[c_size - 1];
  out_pick_j = j_indices[c_size - 1];
  float picked_prob = probs[c_size - 1];

  if (is_stoch) {
    float r = rng.next_float() * sum_prob;
    float running = 0.0f;
    for (int32_t i = 0; i < c_size; ++i) {
      running += probs[i];
      if (r <= running) {
        chosen = candidates[i];
        out_pick_j = j_indices[i];
        picked_prob = probs[i];
        break;
      }
    }
  }

  float log_prob = 0.0f;
  if (is_stoch && sum_prob > EPS) {
    log_prob = std::log(picked_prob / sum_prob);
  }

  return {chosen, is_stoch, log_prob};
}

float MFACO_CVRP::sample_ant_direct_traced(
    const float *probmat, int32_t start_node, std::vector<int32_t> &route_out,
    std::vector<int32_t> &route_raw_out, float &cost_raw_out,
    int32_t &new_edges_out, std::vector<int32_t> &checklist, MFACOTrace &trace,
    Xoshiro128Plus &rng, float &logp_sum, float &survival_out,
    const float *prior) {
  // Copy-paste from sample_ant_direct, with tracing added
  trace.clear();
  trace.start_node = start_node;
  trace.reserve(min_new_edges * 2);

  // 1. Build Adjacency of Source (for new edge detection)
  std::vector<int32_t> src_prev(n, -1);
  std::vector<int32_t> src_next(n, -1);
  std::vector<uint8_t> src_adj_to_depot(n, 0);

  // Original route IDs for cross-route checks
  std::vector<int32_t> source_node_route_id(n, -1);
  int32_t current_route_id = 0;
  for (size_t i = 0; i < source_route.size(); ++i) {
    int32_t u = source_route[i];
    if (u == 0) {
      if (i > 0 && source_route[i - 1] != 0) {
        current_route_id++;
      }
    } else {
      source_node_route_id[u] = current_route_id;
    }
  }

  for (size_t i = 0; i + 1 < source_route.size(); ++i) {
    int32_t u = source_route[i];
    int32_t v = source_route[i + 1];
    if (u > 0 && u < n && v > 0 && v < n) {
      src_next[u] = v;
      src_prev[v] = u;
    } else {
      if (u == 0 && v > 0 && v < n)
        src_adj_to_depot[v] = 1;
      if (v == 0 && u > 0 && u < n)
        src_adj_to_depot[u] = 1;
    }
  }

  auto is_source_edge = [&](int32_t u, int32_t v) -> bool {
    if (u >= n || v >= n)
      return false;
    if (u == 0)
      return src_adj_to_depot[v];
    if (v == 0)
      return src_adj_to_depot[u];
    return (src_next[u] == v || src_prev[u] == v);
  };

  // 2. Initialize Linked List from Source Solution
  // Routes: [0, c1..ck, 0]. In LL: depot_node -> c1 -> ... -> ck -> depot_node
  std::vector<std::vector<int32_t>> init_routes =
      initial_routes_from_perm(source_route);
  int32_t num_routes = (int32_t)init_routes.size();
  int32_t max_routes = n; // Allow growth

  // Structures
  std::vector<int32_t> next_node(n + max_routes);
  std::vector<int32_t> prev_node(n + max_routes);
  std::vector<int32_t> node_route(n, -1); // Only for customers 1..n-1
  std::vector<int64_t> route_loads(max_routes, 0);

  for (int r = 0; r < num_routes; ++r) {
    int32_t depot = n + r;
    int32_t prev = depot;
    int64_t load = 0;

    const auto &rt = init_routes[r];
    // rt is [0, c1...ck, 0]
    for (size_t i = 1; i < rt.size() - 1; ++i) {
      int32_t u = rt[i];
      next_node[prev] = u;
      prev_node[u] = prev;
      node_route[u] = r;
      load += demand_int[u];
      prev = u;
    }
    next_node[prev] = depot;
    prev_node[depot] = prev;
    route_loads[r] = load;
  }

  // 3. Setup Sampling
  std::vector<uint8_t> visited(n, 0);
  int32_t visited_count = 0;

  int32_t curr = start_node;
  if (curr <= 0 || curr >= n) {
    // Robust start node selection
    int attempts = 0;
    while (attempts < 10) {
      curr = 1 + (int32_t)rng.next_uint((uint32_t)m);
      if (curr > 0 && curr < n)
        break;
      attempts++;
    }
    if (curr <= 0 || curr >= n)
      curr = 1;
  }

  visited[curr] = 1;
  visited_count++;

  checklist.clear();
  checklist.push_back(curr);
  std::vector<uint8_t> in_checklist(n, 0);
  in_checklist[curr] = 1;

  int32_t new_edges_all = 0;
  int32_t new_edges_cross = 0;
  int32_t steps = 0;
  int32_t max_steps = m * 4;
  if (fixed_steps > 0)
    max_steps = fixed_steps;

  logp_sum = 0.0f;

  // 4. Main Relocation Loop
  while (true) {
    // Termination Check
    if (fixed_steps > 0) {
      if (steps >= fixed_steps)
        break;
    } else {
      if (new_edges_cross >= min_new_edges || visited_count >= m)
        break;
    }
    if (steps > max_steps)
      break;

    // Determine Current Route info
    int32_t r_curr;
    if (curr >= n) {
      r_curr = curr - n;
    } else {
      r_curr = node_route[curr];
    }

    // Select Next Node 'v'
    int16_t pick_j = -1;
    uint64_t valid_mask = 0;

    // Map depot nodes to 0 for NN/probmat lookups
    int32_t lookup_node = (curr >= n) ? 0 : curr;
    const float *row_prob = probmat + (size_t)lookup_node * k;

    auto [chosen, is_stoch, log_prob] =
        select_next_node(curr, r_curr, row_prob, visited, node_route,
                         route_loads, next_node, prev_node, num_routes,
                         max_routes, rng, pick_j, valid_mask);

    if (chosen == -1)
      break;

    if (is_stoch)
      logp_sum += log_prob;

    // --- Trace Logic ---
    int32_t trace_curr = (curr >= n) ? 0 : curr;
    int32_t trace_chosen = (chosen >= n) ? 0 : chosen;
    bool is_new = !is_source_edge(trace_curr, trace_chosen);

    trace.curr_nodes.push_back(trace_curr);
    trace.chosen_nodes.push_back(trace_chosen);
    trace.is_stochastic.push_back(is_stoch ? 1 : 0);
    trace.pick_j.push_back(pick_j);
    trace.valid_mask.push_back(valid_mask);
    trace.is_new_edge.push_back(is_new ? 1 : 0);
    // -------------------

    // Execute Transition
    // 1. If v == 0: Split / End Route
    if (chosen == 0) {
      int32_t next_c = next_node[curr];

      // If curr is already at end of route (next is depot), just step.
      if (next_c >= n) {
        // Simply traverse to depot
        // Stats
        bool is_cross = true; // depot edge is cross/boundary
        if (is_cross)
          new_edges_cross++; // Count moving to depot as perturbation?
        if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
          new_edges_all++;
        }

        curr = next_c;
        steps++;
        continue;
      }

      // Else: Split.
      if (num_routes >= max_routes) {
        break;
      }

      // Insert new depot after curr
      int32_t r_new = num_routes++;
      int32_t new_depot = n + r_new;
      int32_t old_depot = n + r_curr;

      // Find end of old route (it connects to old_depot)
      int32_t route_end = prev_node[old_depot];

      // Relinking
      next_node[curr] = old_depot;
      prev_node[old_depot] = curr;

      next_node[new_depot] = next_c;
      prev_node[next_c] = new_depot;

      next_node[route_end] = new_depot;
      prev_node[new_depot] = route_end;

      // Update Loads & Route IDs for the new segment
      int64_t load_shift = 0;
      int32_t w = next_c;
      int32_t walk_steps = 0;
      while (w != new_depot && w < n) {
        walk_steps++;
        if (walk_steps > 2 * n)
          break; // safety
        node_route[w] = r_new;
        load_shift += demand_int[w];
        w = next_node[w];
      }
      route_loads[r_curr] -= load_shift;
      route_loads[r_new] = load_shift;

      // Stats for split edge (curr, 0)
      if (!is_source_edge(curr >= n ? 0 : curr, 0)) {
        new_edges_all++;
        if (!in_checklist[curr >= n ? 0 : curr]) {
          checklist.push_back(curr >= n ? 0 : curr);
          in_checklist[curr >= n ? 0 : curr] = 1;
        }
        new_edges_cross++;
      }

      curr = new_depot; // We are now at the start of new route
      steps++;
      continue;
    }

    // 2. If v is Customer: Relocate v after curr
    int32_t v = chosen;
    int32_t r_v = node_route[v];

    // Unlink v
    int32_t prev_v = prev_node[v];
    int32_t next_v = next_node[v];

    if (prev_v == curr) {
      // Already after curr? Just Traverse
      visited[v] = 1;
      visited_count++;
      curr = v;
      steps++;
      continue;
    }

    next_node[prev_v] = next_v;
    prev_node[next_v] = prev_v;

    // Insert v after curr
    int32_t next_c = next_node[curr];
    next_node[curr] = v;
    prev_node[v] = curr;
    next_node[v] = next_c;
    prev_node[next_c] = v;

    // Updates
    if (r_curr != r_v) {
      if (r_v != -1) {
        route_loads[r_v] -= demand_int[v];
      }
      route_loads[r_curr] += demand_int[v];
      node_route[v] = r_curr;
    }

    // Stats
    visited[v] = 1;
    visited_count++;

    int32_t u_idx = (curr >= n) ? 0 : curr;
    if (!is_source_edge(u_idx, v)) {
      new_edges_all++;
      if (!in_checklist[u_idx]) {
        checklist.push_back(u_idx);
        in_checklist[u_idx] = 1;
      }
      if (!in_checklist[v]) {
        checklist.push_back(v);
        in_checklist[v] = 1;
      }

      // Cross check
      bool is_cross = false;
      if (u_idx == 0)
        is_cross = true; // Depot->Node is boundary edge
      else if (source_node_route_id[u_idx] != source_node_route_id[v])
        is_cross = true;

      if (is_cross)
        new_edges_cross++;
    }

    curr = v;
    steps++;
  }

  new_edges_out = new_edges_cross;

  // 5. Flatten Routes
  route_out.clear();
  route_out.reserve(m + num_routes + 1);

  for (int r = 0; r < num_routes; ++r) {
    int32_t d = n + r;
    int32_t w = next_node[d];
    if (w >= n)
      continue; // Empty

    route_out.push_back(0);
    while (w < n) {
      route_out.push_back(w);
      w = next_node[w];
    }
  }
  if (!route_out.empty() && route_out.back() != 0) {
    route_out.push_back(0);
  }

  enforce_time_windows(route_out);

  // Capture raw route before LS
  route_raw_out = route_out;
  cost_raw_out = 0.0f;
  for (size_t i = 0; i + 1 < route_out.size(); ++i) {
    cost_raw_out += dist(route_out[i], route_out[i + 1]);
  }

  // 6. Apply Local Search
  if (use_local_search && !checklist.empty()) {
    std::vector<int32_t> route_before_ls;
    if (has_time_windows)
      route_before_ls = route_out;
    // Intra-Route Or-opt (1/2/3)
    intra_route_oropt(route_out, checklist, 1);
    intra_route_oropt(route_out, checklist, 2);
    intra_route_oropt(route_out, checklist, 3);
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
    // Inter-Route LS
    std::vector<int32_t> pos_ls(n, -1);
    inter_route_ls_optimized(route_out, pos_ls, checklist, in_checklist);
    // Intra-Route Or-opt (1/2/3)
    intra_route_oropt(route_out, checklist, 1);
    intra_route_oropt(route_out, checklist, 2);
    intra_route_oropt(route_out, checklist, 3);
    // Intra-Route LS (2-opt)
    intra_route_ls(route_out, checklist);
    if (has_time_windows && !route_fully_feasible(route_out))
      route_out.swap(route_before_ls);
  }
  if (has_time_windows && !route_fully_feasible(route_out)) {
    enforce_time_windows(route_out);
    if (!route_fully_feasible(route_out))
      route_out = source_route;
  }

  // Calculate final cost
  float final_cost = 0.0f;
  for (size_t i = 0; i + 1 < route_out.size(); ++i) {
    final_cost += dist(route_out[i], route_out[i + 1]);
  }

  return final_cost;
}

} // namespace mfaco
