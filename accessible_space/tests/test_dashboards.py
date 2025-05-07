import pytest

try:
    import accessible_space.apps.readme
    import accessible_space.apps.validation
    import accessible_space.apps.qualitative_profiling
except ImportError as e:
    pytest.skip(f"Skipping tests because import failed: {e}", allow_module_level=True)


def test_readme_dashboard():
    accessible_space.apps.readme.main(run_as_streamlit_app=False)


def test_qualitative_profiling_dashboard():
    accessible_space.apps.qualitative_profiling.parameter_exploration_dashboard()


def test_validation_dashboard():
    df_test_scores, biggest_xc_in_test_set, avg_xc_total_only_success_test, avg_xc_total_only_failure_test, target_density_success, target_density_fail = accessible_space.apps.validation.main(run_as_streamlit_app=False, dummy=False, run_asserts=True)
    top_result = df_test_scores.iloc[0]

    def _assert(a, b):
        assert a == b

    # assert round(target_density_fail, 3) == 0.427
    # assert round(target_density_success, 3) == 0.882
    # assert round(biggest_xc_in_test_set, 3) == 0.982
    # assert round(avg_xc_total_only_success_test, 3) == 0.852
    #
    # _assert(round(top_result["logloss"], 3), 0.243)
    # _assert(round(top_result["logloss_real"], 3), 0.387),
    # _assert(round(top_result["brier_score"], 3), 0.075),
    # _assert(round(top_result["brier_score_real"], 3), 0.119),
    # _assert(round(top_result["ece"], 3), 0.029),
    # _assert(round(top_result["auc"], 3), 0.959),
    #
    # _assert(round(top_result["logloss_ci_lower"], 3), 0.218),
    # _assert(round(top_result["logloss_ci_upper"], 3), 0.269),
    # _assert(round(top_result["brier_ci_lower"], 3), 0.066),
    # _assert(round(top_result["brier_ci_upper"], 3), 0.083),
    # _assert(round(top_result["auc_ci_lower"], 3), 0.950),
    # _assert(round(top_result["auc_ci_upper"], 3), 0.967),
    # _assert(round(top_result["ece_ci_lower"], 3), 0.021),
    # _assert(round(top_result["ece_ci_upper"], 3), 0.042),
    #
    # _assert(round(top_result["logloss_real"], 3), 0.387),
    # _assert(round(top_result["brier_score_real"], 3), 0.119),
    # _assert(round(top_result["auc_real"], 3), 0.832),
    # _assert(round(top_result["logloss_ci_lower_real"], 3), 0.350),
    # _assert(round(top_result["logloss_ci_upper_real"], 3), 0.424),
    # _assert(round(top_result["brier_ci_lower_real"], 3), 0.106),
    # _assert(round(top_result["brier_ci_upper_real"], 3), 0.134),
    # _assert(round(top_result["auc_ci_lower_real"], 3), 0.800),
    # _assert(round(top_result["auc_ci_upper_real"], 3), 0.861),
    #
    # _assert(round(top_result["baseline_logloss"], 3), 0.693),
    # _assert(round(top_result["baseline_brier"], 3), 0.250),
    # _assert(round(top_result["baseline_auc"], 3), 0.500),
    # _assert(round(top_result["baseline_loglos_real"], 3), 0.494),
    # _assert(round(top_result["baseline_brier_real"], 3), 0.157),
    # _assert(round(top_result["baseline_auc_real"], 3), 0.500),
