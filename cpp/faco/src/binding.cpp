/**
 * pybind11 bindings for Unified MFACO Training Module
 *
 * Exposes MFACO_CVRP, MFACO_CVRPTW, and MFACOTrace to Python.
 * Module name: faco_opt
 */

#include "mfaco_train.h"
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mfaco;

// ============================================================================
// Helper: create numpy array view from vector (no copy)
// ============================================================================

template <typename T>
py::array_t<T> make_view(T *data, std::vector<py::ssize_t> shape) {
  // Compute strides (row-major)
  std::vector<py::ssize_t> strides(shape.size());
  py::ssize_t stride = sizeof(T);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return py::array_t<T>(shape, strides, data, py::none());
}

template <typename T> py::array_t<T> make_1d_view(T *data, py::ssize_t len) {
  return make_view<T>(data, {len});
}

template <typename T>
py::array_t<T> make_2d_view(T *data, py::ssize_t rows, py::ssize_t cols) {
  return make_view<T>(data, {rows, cols});
}

// ============================================================================
// MFACOTraceBatch Python wrapper (Shared)
// ============================================================================

class PyMFACOTrace {
public:
  // The trace batch from C++
  MFACOTraceBatch batch;
  int32_t n_ants;

  PyMFACOTrace() : n_ants(0) {}

  // Property accessors returning numpy views
  py::array_t<int32_t> get_starts() {
    return make_1d_view(batch.starts.data(), batch.starts.size());
  }

  py::array_t<int32_t> get_curr_nodes() {
    return make_1d_view(batch.curr_nodes.data(), batch.curr_nodes.size());
  }

  py::array_t<int32_t> get_chosen_nodes() {
    return make_1d_view(batch.chosen_nodes.data(), batch.chosen_nodes.size());
  }

  py::array_t<uint8_t> get_is_stochastic() {
    return make_1d_view(batch.is_stochastic.data(), batch.is_stochastic.size());
  }

  py::array_t<int16_t> get_pick_j() {
    return make_1d_view(batch.pick_j.data(), batch.pick_j.size());
  }

  py::array_t<uint64_t> get_valid_mask() {
    return make_1d_view(batch.valid_mask.data(), batch.valid_mask.size());
  }

  py::array_t<uint8_t> get_is_new_edge() {
    return make_1d_view(batch.is_new_edge.data(), batch.is_new_edge.size());
  }

  py::array_t<int32_t> get_start_nodes() {
    return make_1d_view(batch.start_nodes.data(), batch.start_nodes.size());
  }

  int32_t n_decisions() const {
    return static_cast<int32_t>(batch.curr_nodes.size());
  }

  // Convert batch to list of Python-compatible trace dicts for
  // replay_logp_batch
  py::list to_trace_list() {
    py::list result;
    for (int32_t a = 0; a < n_ants; ++a) {
      py::dict trace;
      trace["start_node"] = batch.start_nodes[a];

      int32_t start = batch.starts[a];
      int32_t end = batch.starts[a + 1];

      py::list curr, chosen, is_stoch;
      for (int32_t i = start; i < end; ++i) {
        curr.append(batch.curr_nodes[i]);
        chosen.append(batch.chosen_nodes[i]);
        is_stoch.append(batch.is_stochastic[i] != 0);
      }

      trace["curr_nodes"] = curr;
      trace["chosen_nodes"] = chosen;
      trace["is_stochastic"] = is_stoch;
      trace["is_new_edge"] =
          make_1d_view(batch.is_new_edge.data() + start, end - start);

      result.append(trace);
    }
    return result;
  }
};

// ============================================================================
// MFACO_CVRP Python wrapper
// ============================================================================

class PyMFACO_CVRP {
public:
  std::unique_ptr<MFACO_CVRP> solver;

  PyMFACO_CVRP(py::array_t<float, py::array::c_style | py::array::forcecast>
                   coords, // (n,2)
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   demand, // (n,)
               float capacity, int32_t n_ants, int32_t cand_list_size = 32,
               int32_t backup_list_size = 32, int32_t min_new_edges = 8,
               float decay = 0.9f, float alpha = 1.0f, float p_best = 0.05f,
               bool use_local_search = true, bool disable_heuristic = false,

               bool extend_ls = false, bool smooth_mmas = false,
               int32_t fixed_steps = 0, bool nls = false, int32_t T_nls = 10,
               bool deep_nls = false) {
    auto cbuf = coords.request();
    if (cbuf.ndim != 2 || cbuf.shape[1] != 2) {
      throw std::runtime_error("coords must be shape (n,2)");
    }
    int32_t n = (int32_t)cbuf.shape[0];

    auto dbuf = demand.request();
    if (dbuf.ndim != 1 || (int32_t)dbuf.shape[0] != n) {
      throw std::runtime_error("demand must be shape (n,) matching coords");
    }

    solver = std::make_unique<MFACO_CVRP>(
        (const float *)cbuf.ptr, (const float *)dbuf.ptr, n, capacity, n_ants,
        cand_list_size, backup_list_size, min_new_edges, decay, alpha, p_best,
        use_local_search, disable_heuristic, extend_ls, smooth_mmas,
        fixed_steps, nls, T_nls, deep_nls);
  }

  // properties
  int32_t get_n() const { return solver->n; }
  int32_t get_m() const { return solver->m; }
  int32_t get_n_ants() const { return solver->n_ants; }
  int32_t get_k() const { return solver->k; }
  int32_t get_bl() const { return solver->bl; }
  float get_source_cost() const { return solver->source_cost; }
  float get_best_cost() const { return solver->best_cost; }
  float get_tau_min() const { return solver->tau_min; }
  float get_tau_max() const { return solver->tau_max; }

  py::array_t<float> get_pheromone_sparse_np() {
    return make_2d_view(solver->pheromone_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_nn_list() {
    return make_2d_view(solver->nn_list_data(), solver->n, solver->k);
  }
  py::array_t<int32_t> get_backup_list() {
    return make_2d_view(solver->backup_list_data(), solver->n, solver->bl);
  }
  py::array_t<float> get_heuristic_sparse_np() {
    return make_2d_view(solver->heuristic_data(), solver->n, solver->k);
  }
  // py::array_t<int32_t> get_nn_pos() { ... } REMOVED

  py::array_t<int32_t> get_source_route() {
    return make_1d_view(solver->source_route.data(),
                        solver->source_route.size());
  }
  py::array_t<int32_t> get_best_route() {
    return make_1d_view(solver->best_route.data(), solver->best_route.size());
  }

  void seed_rng(uint64_t seed) { solver->seed_rng(seed); }

  py::tuple sample(bool require_prob = false, py::object prior = py::none(),
                   bool parallel_traced = false, bool return_decoded = false) {
    const float *prior_ptr = nullptr;
    py::array_t<float> prior_arr;

    if (!prior.is_none()) {
      prior_arr = prior.cast<
          py::array_t<float, py::array::c_style | py::array::forcecast>>();
      auto rbuf = prior_arr.request();
      if (rbuf.ndim != 2 || rbuf.shape[0] != solver->n ||
          rbuf.shape[1] != solver->k) {
        throw std::runtime_error("prior must be shape (n,k)");
      }
      prior_ptr = (const float *)rbuf.ptr;
    }

    SampleResult result;
    {
      py::gil_scoped_release release;
      solver->sample(require_prob, prior_ptr, result, parallel_traced);
    }

    py::array_t<float> costs(solver->n_ants);
    auto cb = costs.mutable_unchecked<1>();
    for (int32_t a = 0; a < solver->n_ants; ++a)
      cb(a) = result.costs[a];

    py::list routes;
    for (int32_t a = 0; a < solver->n_ants; ++a) {
      const auto &r = result.routes[a];
      py::array_t<int32_t> r_arr((py::ssize_t)r.size());
      auto rb = r_arr.mutable_unchecked<1>();
      for (size_t i = 0; i < r.size(); ++i)
        rb(i) = r[i];
      routes.append(r_arr);
    }

    py::object decoded_obj = py::none();
    if (return_decoded) {
      decoded_obj = routes;
    }

    py::object traces_obj = py::none();
    if (require_prob) {
      auto t = std::make_unique<PyMFACOTrace>();
      t->batch = std::move(result.traces);
      t->n_ants = solver->n_ants;
      traces_obj = py::cast(std::move(t));
    }

    py::array_t<float> costs_raw(solver->n_ants);
    if (!result.costs_raw.empty()) {
      auto cb = costs_raw.mutable_unchecked<1>();
      for (int32_t a = 0; a < solver->n_ants; ++a)
        cb(a) = result.costs_raw[a];
    }

    py::list perms_raw;
    if (!result.routes_raw.empty()) {
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        if (!result.routes_raw[a].empty()) {
          const auto &r = result.routes_raw[a];
          py::array_t<int32_t> r_arr((py::ssize_t)r.size());
          auto rb = r_arr.mutable_unchecked<1>();
          for (size_t i = 0; i < r.size(); ++i)
            rb(i) = r[i];
          perms_raw.append(r_arr);
        } else {
          perms_raw.append(py::none());
        }
      }
    }

    py::array_t<float> logps_arr(solver->n_ants);
    if (!result.logps.empty()) {
      auto logps_buf = logps_arr.mutable_unchecked<1>();
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        logps_buf(a) = result.logps[a];
      }
    }

    py::array_t<int32_t> new_edges_arr(solver->n_ants);
    auto ne_buf = new_edges_arr.mutable_unchecked<1>();
    if (!result.new_edges_count.empty()) {
      for (int32_t a = 0; a < solver->n_ants; ++a) {
        ne_buf(a) = result.new_edges_count[a];
      }
    } else {
      for (int32_t a = 0; a < solver->n_ants; ++a)
        ne_buf(a) = 0;
    }

    py::array_t<float> survival_arr(solver->n_ants);
    auto surv_buf = survival_arr.mutable_unchecked<1>();
    if (!result.edge_survival.empty()) {
      for (int32_t a = 0; a < solver->n_ants; ++a)
        surv_buf(a) = result.edge_survival[a];
    } else {
      for (int32_t a = 0; a < solver->n_ants; ++a)
        surv_buf(a) = 0.0f;
    }

    return py::make_tuple(costs, routes, decoded_obj, logps_arr, traces_obj,
                          costs_raw, perms_raw, new_edges_arr, survival_arr);
  }

  void update_pheromone_from_route(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast>
          best_route,
      float best_cost) {
    auto buf = best_route.request();
    if (buf.ndim != 1) {
      throw std::runtime_error("best_route must be 1D");
    }
    const int32_t *p = (const int32_t *)buf.ptr;
    std::vector<int32_t> route_vec(p, p + buf.shape[0]);
    py::gil_scoped_release release;
    solver->update_pheromone(route_vec, best_cost);
  }

  bool set_source_route(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> route,
      float cost) {
    auto buf = route.request();
    if (buf.ndim != 1) {
      throw std::runtime_error("route must be 1D");
    }
    const int32_t *p = (const int32_t *)buf.ptr;
    std::vector<int32_t> route_vec(p, p + buf.shape[0]);
    py::gil_scoped_release release;
    return solver->set_source_route(route_vec, cost);
  }

  void reset_timings() { solver->reset_timings(); }
  py::dict get_timings() {
    py::dict d;
    d["time_ant"] = solver->time_ant;
    d["time_ls"] = solver->time_ls;
    d["time_split"] = solver->time_split;
    d["time_inter_ls"] = solver->time_inter_ls;
    d["time_intra_ls"] = solver->time_intra_ls;
    d["time_deep_ls"] = solver->time_deep_ls;
    d["count_inter_ls"] = solver->count_inter_ls;
    d["count_intra_ls"] = solver->count_intra_ls;
    d["count_deep_ls"] = solver->count_deep_ls;
    d["fts_checks"] = solver->fts_checks;
    d["fts_fallback_scans"] = solver->fts_fallback_scans;
    return d;
  }
};

class PyMFACO_CVRPTW : public PyMFACO_CVRP {
public:
  PyMFACO_CVRPTW(
      py::array_t<float, py::array::c_style | py::array::forcecast> coords,
      py::array_t<float, py::array::c_style | py::array::forcecast> demand,
      py::array_t<float, py::array::c_style | py::array::forcecast> windows,
      float capacity, int32_t n_ants, int32_t cand_list_size = 32,
      int32_t backup_list_size = 32, int32_t min_new_edges = 8,
      float decay = 0.9f, float alpha = 1.0f, float p_best = 0.05f,
      bool use_local_search = true, bool disable_heuristic = false,
      bool extend_ls = false, bool smooth_mmas = false,
      int32_t fixed_steps = 0, bool nls = false, int32_t T_nls = 10,
      bool deep_nls = false, int32_t granular_mode = 0, float granular_wait_weight = 0.2f,
      float granular_time_warp_weight = 1.0f, bool hgs_soft_deep_ls = false,
      bool hgs_soft_cheap_ls = false, bool hgs_soft_intra_ls = false,
      bool hgs_deep_ls = false, int32_t hgs_deep_top_k = 0,
      float hgs_tw_penalty = 10.0f,
      float hgs_capacity_penalty = 10.0f, bool hgs_adaptive_penalty = false,
      float hgs_target_feasible = 0.8f, int32_t hgs_deep_rounds = 1,
      bool hgs_deep_route_pair_prune = false,
      int32_t hgs_deep_route_pair_top_k = 3)
      : PyMFACO_CVRP(coords, demand, capacity, n_ants, cand_list_size,
                     backup_list_size, min_new_edges, decay, alpha, p_best,
                     use_local_search, disable_heuristic, extend_ls,
                     smooth_mmas, fixed_steps, nls, T_nls, deep_nls) {
    auto cbuf = coords.request();
    auto wbuf = windows.request();
    if (wbuf.ndim != 2 || wbuf.shape[0] != cbuf.shape[0] ||
        wbuf.shape[1] != 2) {
      throw std::runtime_error("windows must be shape (n,2) matching coords");
    }
    solver->granular_mode = granular_mode;
    solver->granular_wait_weight = granular_wait_weight;
    solver->granular_time_warp_weight = granular_time_warp_weight;
    solver->hgs_soft_deep_ls = hgs_soft_deep_ls;
    solver->hgs_soft_cheap_ls = hgs_soft_cheap_ls;
    solver->hgs_soft_intra_ls = hgs_soft_intra_ls;
    solver->hgs_deep_ls = hgs_deep_ls;
    solver->hgs_deep_top_k = hgs_deep_top_k;
    solver->hgs_tw_penalty = hgs_tw_penalty;
    solver->hgs_capacity_penalty = hgs_capacity_penalty;
    solver->hgs_tw_penalty_current = hgs_tw_penalty;
    solver->hgs_capacity_penalty_current = hgs_capacity_penalty;
    solver->hgs_adaptive_penalty = hgs_adaptive_penalty;
    solver->hgs_target_feasible = hgs_target_feasible;
    solver->hgs_deep_rounds = hgs_deep_rounds;
    solver->hgs_deep_route_pair_prune = hgs_deep_route_pair_prune;
    solver->hgs_deep_route_pair_top_k = hgs_deep_route_pair_top_k;
    solver->set_time_windows((const float *)wbuf.ptr);
  }
};

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(faco_opt, m) {
  m.doc() = "Unified C++ MFACO Training Module for TSP and CVRP";

  // OpenMP controls
  m.def(
      "set_num_threads",
      [](int n_threads) {
        if (n_threads <= 0)
          throw std::runtime_error("n_threads must be > 0");
        omp_set_num_threads(n_threads);
      },
      py::arg("n_threads"), "Set OpenMP thread count");

  m.def("get_max_threads", []() { return omp_get_max_threads(); });
  m.def("get_num_procs", []() { return omp_get_num_procs(); });
  m.def(
      "set_dynamic", [](bool enabled) { omp_set_dynamic(enabled ? 1 : 0); },
      py::arg("enabled"));
  m.def("get_dynamic", []() { return omp_get_dynamic() != 0; });

  // MFACOTrace
  py::class_<PyMFACOTrace>(m, "MFACOTrace")
      .def(py::init<>())
      .def_property_readonly("starts", &PyMFACOTrace::get_starts)
      .def_property_readonly("curr_nodes", &PyMFACOTrace::get_curr_nodes)
      .def_property_readonly("chosen_nodes", &PyMFACOTrace::get_chosen_nodes)
      .def_property_readonly("is_stochastic", &PyMFACOTrace::get_is_stochastic)
      .def_property_readonly("pick_j", &PyMFACOTrace::get_pick_j)
      .def_property_readonly("valid_mask", &PyMFACOTrace::get_valid_mask)
      .def_property_readonly("is_new_edge", &PyMFACOTrace::get_is_new_edge)
      .def_property_readonly("start_nodes", &PyMFACOTrace::get_start_nodes)
      .def_property_readonly("n_decisions", &PyMFACOTrace::n_decisions)
      .def_property_readonly("n_ants",
                             [](const PyMFACOTrace &t) { return t.n_ants; })
      .def("to_trace_list", &PyMFACOTrace::to_trace_list);

  // MFACO_CVRP
  py::class_<PyMFACO_CVRP>(m, "MFACO_CVRP")
      .def(py::init<
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               float, int32_t, int32_t, int32_t, int32_t, float, float, float,
               bool, bool, bool, bool, int32_t, bool, int32_t, bool>(),
           py::arg("coords"), py::arg("demand"), py::arg("capacity"),
           py::arg("n_ants"), py::arg("cand_list_size") = 32,
           py::arg("backup_list_size") = 32, py::arg("min_new_edges") = 8,
           py::arg("decay") = 0.9f, py::arg("alpha") = 1.0f,
           py::arg("p_best") = 0.05f, py::arg("use_local_search") = true,
           py::arg("disable_heuristic") = false, py::arg("extend_ls") = false,
           py::arg("smooth_mmas") = false, py::arg("fixed_steps") = 0,
           py::arg("nls") = false, py::arg("T_nls") = 10,
           py::arg("deep_nls") = false)
      .def_property_readonly("n", &PyMFACO_CVRP::get_n)
      .def_property_readonly("m", &PyMFACO_CVRP::get_m)
      .def_property_readonly("n_ants", &PyMFACO_CVRP::get_n_ants)
      .def_property_readonly("k", &PyMFACO_CVRP::get_k)
      .def_property_readonly("bl", &PyMFACO_CVRP::get_bl)
      .def_property_readonly("source_cost", &PyMFACO_CVRP::get_source_cost)
      .def_property_readonly("best_cost", &PyMFACO_CVRP::get_best_cost)
      .def_property_readonly("tau_min", &PyMFACO_CVRP::get_tau_min)
      .def_property_readonly("tau_max", &PyMFACO_CVRP::get_tau_max)
      .def_property_readonly("pheromone_sparse_np",
                             &PyMFACO_CVRP::get_pheromone_sparse_np)
      .def_property_readonly("nn_list", &PyMFACO_CVRP::get_nn_list)
      .def_property_readonly("backup_list", &PyMFACO_CVRP::get_backup_list)
      .def_property_readonly("heuristic_sparse_np",
                             &PyMFACO_CVRP::get_heuristic_sparse_np)
      .def_property_readonly("heuristic_sparse_np",
                             &PyMFACO_CVRP::get_heuristic_sparse_np)
      // .def_property_readonly("nn_pos", &PyMFACO_CVRP::get_nn_pos)
      .def_property_readonly("source_route", &PyMFACO_CVRP::get_source_route)
      .def_property_readonly("best_route", &PyMFACO_CVRP::get_best_route)
      .def("seed_rng", &PyMFACO_CVRP::seed_rng)
      .def("sample", &PyMFACO_CVRP::sample, py::arg("require_prob") = false,
           py::arg("prior") = py::none(), py::arg("parallel_traced") = false,
           py::arg("return_decoded") = false)
      .def("update_pheromone_from_route",
           &PyMFACO_CVRP::update_pheromone_from_route)
      .def("set_source_route", &PyMFACO_CVRP::set_source_route)
      .def_property(
          "use_relocate",
          [](PyMFACO_CVRP &self) { return self.solver->use_relocate; },
          [](PyMFACO_CVRP &self, bool v) { self.solver->use_relocate = v; })
      .def_property(
          "use_swap", [](PyMFACO_CVRP &self) { return self.solver->use_swap; },
          [](PyMFACO_CVRP &self, bool v) { self.solver->use_swap = v; })
      .def_property(
          "use_2opt_star",
          [](PyMFACO_CVRP &self) { return self.solver->use_2opt_star; },
          [](PyMFACO_CVRP &self, bool v) { self.solver->use_2opt_star = v; })
      .def_property(
          "use_fts_checks",
          [](PyMFACO_CVRP &self) { return self.solver->use_fts_checks; },
          [](PyMFACO_CVRP &self, bool v) { self.solver->use_fts_checks = v; })
      .def("reset_timings", &PyMFACO_CVRP::reset_timings)
      .def("get_timings", &PyMFACO_CVRP::get_timings);

  py::class_<PyMFACO_CVRPTW, PyMFACO_CVRP>(m, "MFACO_CVRPTW")
      .def(py::init<
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               py::array_t<float, py::array::c_style | py::array::forcecast>,
               float, int32_t, int32_t, int32_t, int32_t, float, float, float,
               bool, bool, bool, bool, int32_t, bool, int32_t, bool, int32_t, float,
               float, bool, bool, bool, bool, int32_t, float, float, bool, float, int32_t, bool, int32_t>(),
           py::arg("coords"), py::arg("demand"), py::arg("windows"),
           py::arg("capacity"), py::arg("n_ants"),
           py::arg("cand_list_size") = 32, py::arg("backup_list_size") = 32,
           py::arg("min_new_edges") = 8, py::arg("decay") = 0.9f,
           py::arg("alpha") = 1.0f, py::arg("p_best") = 0.05f,
           py::arg("use_local_search") = true,
           py::arg("disable_heuristic") = false,
           py::arg("extend_ls") = false, py::arg("smooth_mmas") = false,
           py::arg("fixed_steps") = 0, py::arg("nls") = false,
           py::arg("T_nls") = 10, py::arg("deep_nls") = false,
           py::arg("granular_mode") = 0,
           py::arg("granular_wait_weight") = 0.2f,
           py::arg("granular_time_warp_weight") = 1.0f,
           py::arg("hgs_soft_deep_ls") = false,
           py::arg("hgs_soft_cheap_ls") = false,
           py::arg("hgs_soft_intra_ls") = false,
           py::arg("hgs_deep_ls") = false,
           py::arg("hgs_deep_top_k") = 0,
           py::arg("hgs_tw_penalty") = 10.0f,
           py::arg("hgs_capacity_penalty") = 10.0f,
           py::arg("hgs_adaptive_penalty") = false,
           py::arg("hgs_target_feasible") = 0.8f,
           py::arg("hgs_deep_rounds") = 1,
           py::arg("hgs_deep_route_pair_prune") = false,
           py::arg("hgs_deep_route_pair_top_k") = 3)
      .def_property_readonly("n", &PyMFACO_CVRPTW::get_n)
      .def_property_readonly("m", &PyMFACO_CVRPTW::get_m)
      .def_property_readonly("n_ants", &PyMFACO_CVRPTW::get_n_ants)
      .def_property_readonly("k", &PyMFACO_CVRPTW::get_k)
      .def_property_readonly("bl", &PyMFACO_CVRPTW::get_bl)
      .def_property_readonly("source_cost", &PyMFACO_CVRPTW::get_source_cost)
      .def_property_readonly("best_cost", &PyMFACO_CVRPTW::get_best_cost)
      .def_property_readonly("tau_min", &PyMFACO_CVRPTW::get_tau_min)
      .def_property_readonly("tau_max", &PyMFACO_CVRPTW::get_tau_max)
      .def_property_readonly("pheromone_sparse_np",
                             &PyMFACO_CVRPTW::get_pheromone_sparse_np)
      .def_property_readonly("nn_list", &PyMFACO_CVRPTW::get_nn_list)
      .def_property_readonly("backup_list", &PyMFACO_CVRPTW::get_backup_list)
      .def_property_readonly("heuristic_sparse_np",
                             &PyMFACO_CVRPTW::get_heuristic_sparse_np)
      .def_property_readonly("source_route", &PyMFACO_CVRPTW::get_source_route)
      .def_property_readonly("best_route", &PyMFACO_CVRPTW::get_best_route)
      .def("seed_rng", &PyMFACO_CVRPTW::seed_rng)
      .def("sample", &PyMFACO_CVRPTW::sample, py::arg("require_prob") = false,
           py::arg("prior") = py::none(), py::arg("parallel_traced") = false,
           py::arg("return_decoded") = false)
      .def("update_pheromone_from_route",
           &PyMFACO_CVRPTW::update_pheromone_from_route)
      .def("set_source_route", &PyMFACO_CVRPTW::set_source_route)
      .def_property(
          "use_relocate",
          [](PyMFACO_CVRPTW &self) { return self.solver->use_relocate; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->use_relocate = v; })
      .def_property(
          "use_swap", [](PyMFACO_CVRPTW &self) { return self.solver->use_swap; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->use_swap = v; })
      .def_property(
          "use_2opt_star",
          [](PyMFACO_CVRPTW &self) { return self.solver->use_2opt_star; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->use_2opt_star = v; })
      .def_property(
          "use_fts_checks",
          [](PyMFACO_CVRPTW &self) { return self.solver->use_fts_checks; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->use_fts_checks = v; })
      .def_property(
          "hgs_soft_deep_ls",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_soft_deep_ls; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->hgs_soft_deep_ls = v; })
      .def_property(
          "hgs_soft_cheap_ls",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_soft_cheap_ls; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->hgs_soft_cheap_ls = v; })
      .def_property(
          "hgs_soft_intra_ls",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_soft_intra_ls; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->hgs_soft_intra_ls = v; })
      .def_property(
          "hgs_deep_ls",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_deep_ls; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->hgs_deep_ls = v; })
      .def_property(
          "hgs_deep_top_k",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_deep_top_k; },
          [](PyMFACO_CVRPTW &self, int32_t v) { self.solver->hgs_deep_top_k = v; })
      .def_property(
          "hgs_tw_penalty",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_tw_penalty; },
          [](PyMFACO_CVRPTW &self, float v) {
            self.solver->hgs_tw_penalty = v;
            self.solver->hgs_tw_penalty_current = v;
          })
      .def_property(
          "hgs_capacity_penalty",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_capacity_penalty; },
          [](PyMFACO_CVRPTW &self, float v) {
            self.solver->hgs_capacity_penalty = v;
            self.solver->hgs_capacity_penalty_current = v;
          })
      .def_property(
          "hgs_adaptive_penalty",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_adaptive_penalty; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->hgs_adaptive_penalty = v; })
      .def_property(
          "hgs_target_feasible",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_target_feasible; },
          [](PyMFACO_CVRPTW &self, float v) { self.solver->hgs_target_feasible = v; })
      .def_property(
          "hgs_deep_rounds",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_deep_rounds; },
          [](PyMFACO_CVRPTW &self, int32_t v) { self.solver->hgs_deep_rounds = v; })
      .def_property(
          "hgs_deep_route_pair_prune",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_deep_route_pair_prune; },
          [](PyMFACO_CVRPTW &self, bool v) { self.solver->hgs_deep_route_pair_prune = v; })
      .def_property(
          "hgs_deep_route_pair_top_k",
          [](PyMFACO_CVRPTW &self) { return self.solver->hgs_deep_route_pair_top_k; },
          [](PyMFACO_CVRPTW &self, int32_t v) { self.solver->hgs_deep_route_pair_top_k = v; })
      .def("reset_timings", &PyMFACO_CVRPTW::reset_timings)
      .def("get_timings", &PyMFACO_CVRPTW::get_timings);


}
