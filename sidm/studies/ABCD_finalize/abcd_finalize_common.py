"""ABCD finalization helpers consolidated by responsibility.

The source sections retain the original algorithms and guards. Static, prefixed
global names isolate study-specific settings; namespaces expose the original
APIs without loading deleted modules or embedding executable source strings.
"""

from importlib import import_module as _finalize_import_module


class _FinalizeNamespace:
    """Forward a study API to its actual implementation, including rebinding."""
    def __init__(self, module, bindings):
        object.__setattr__(self, '_module', module)
        object.__setattr__(self, '_bindings', bindings)

    def __getattr__(self, name):
        module = _finalize_import_module(self._module)
        if name in ('__file__', '__name__'):
            return getattr(module, name)
        target = self._bindings.get(name)
        if target is None:
            raise AttributeError(name)
        return getattr(module, target)

    def __setattr__(self, name, value):
        target = self._bindings.get(name)
        if target is None:
            raise AttributeError(name)
        setattr(_finalize_import_module(self._module), target, value)


_FINALIZE_NAMESPACES = {}
_FINALIZE_NAMESPACES['abcd_4mu_final_09style'] = _FinalizeNamespace('abcd_finalize_plots', {'Path': 'finalize_4mu_final_09style_Path', 'pd': 'finalize_4mu_final_09style_pd', 'reference_presentation_order': 'finalize_4mu_final_09style_reference_presentation_order', 'make_09_style_figures': 'finalize_4mu_final_09style_make_09_style_figures'})
_FINALIZE_NAMESPACES['abcd_4mu_final_09style_scans'] = _FinalizeNamespace('abcd_finalize_plots', {'np': 'finalize_4mu_final_09style_scans_np', 'pd': 'finalize_4mu_final_09style_scans_pd', 'plt': 'finalize_4mu_final_09style_scans_plt', 'Line2D': 'finalize_4mu_final_09style_scans_Line2D', 'Rectangle': 'finalize_4mu_final_09style_scans_Rectangle', 'reference': 'finalize_4mu_final_09style_scans_reference', 'gap_reference': 'finalize_4mu_final_09style_scans_gap_reference', 'comparison_reference': 'finalize_4mu_final_09style_scans_comparison_reference', 'reference_canvas': 'finalize_4mu_final_09style_scans_reference_canvas', '_caption': 'finalize_4mu_final_09style_scans__caption', '_style_table': 'finalize_4mu_final_09style_scans__style_table', 'ANGLES': 'finalize_4mu_final_09style_scans_ANGLES', 'COLORS': 'finalize_4mu_final_09style_scans_COLORS', 'METRICS': 'finalize_4mu_final_09style_scans_METRICS', 'COMPARISON_LABEL': 'finalize_4mu_final_09style_scans_COMPARISON_LABEL', 'BUFFER_COMPARISON_LABEL': 'finalize_4mu_final_09style_scans_BUFFER_COMPARISON_LABEL', 'make_scan_figures': 'finalize_4mu_final_09style_scans_make_scan_figures', 'display_mc_plane': 'finalize_4mu_final_09style_scans_display_mc_plane', '_inputs': 'finalize_4mu_final_09style_scans__inputs', '_canvas': 'finalize_4mu_final_09style_scans__canvas', '_channel': 'finalize_4mu_final_09style_scans__channel', '_replace_caption': 'finalize_4mu_final_09style_scans__replace_caption', '_nominal_plane': 'finalize_4mu_final_09style_scans__nominal_plane', '_approaches': 'finalize_4mu_final_09style_scans__approaches', '_tradeoff': 'finalize_4mu_final_09style_scans__tradeoff', '_selection': 'finalize_4mu_final_09style_scans__selection', '_gap_tradeoff': 'finalize_4mu_final_09style_scans__gap_tradeoff', '_nominal_distribution': 'finalize_4mu_final_09style_scans__nominal_distribution', '_gap_closure': 'finalize_4mu_final_09style_scans__gap_closure', '_gap_selection': 'finalize_4mu_final_09style_scans__gap_selection', '_nominal_contamination': 'finalize_4mu_final_09style_scans__nominal_contamination', '_comparison_signals': 'finalize_4mu_final_09style_scans__comparison_signals', '_meeting_comparison': 'finalize_4mu_final_09style_scans__meeting_comparison', '_unavailable_data_significance': 'finalize_4mu_final_09style_scans__unavailable_data_significance'})
_FINALIZE_NAMESPACES['abcd_4mu_final_09style_validation'] = _FinalizeNamespace('abcd_finalize_plots', {'hashlib': 'finalize_4mu_final_09style_validation_hashlib', 'np': 'finalize_4mu_final_09style_validation_np', 'pd': 'finalize_4mu_final_09style_validation_pd', 'HandlerTuple': 'finalize_4mu_final_09style_validation_HandlerTuple', 'Line2D': 'finalize_4mu_final_09style_validation_Line2D', 'style09': 'finalize_4mu_final_09style_validation_style09', '_canvas': 'finalize_4mu_final_09style_validation__canvas', '_caption': 'finalize_4mu_final_09style_validation__caption', '_intermediate_overlay': 'finalize_4mu_final_09style_validation__intermediate_overlay', 'MASSES': 'finalize_4mu_final_09style_validation_MASSES', 'STATES': 'finalize_4mu_final_09style_validation_STATES', 'MASS_LABELS': 'finalize_4mu_final_09style_validation_MASS_LABELS', 'SCOPE_PREFIX': 'finalize_4mu_final_09style_validation_SCOPE_PREFIX', 'DATA_FIELDS': 'finalize_4mu_final_09style_validation_DATA_FIELDS', 'make_validation_figures': 'finalize_4mu_final_09style_validation_make_validation_figures', '_signature': 'finalize_4mu_final_09style_validation__signature', '_bool': 'finalize_4mu_final_09style_validation__bool', '_alias_scope': 'finalize_4mu_final_09style_validation__alias_scope', '_render_adapter': 'finalize_4mu_final_09style_validation__render_adapter', '_validate_data_source': 'finalize_4mu_final_09style_validation__validate_data_source', '_two_dimensional': 'finalize_4mu_final_09style_validation__two_dimensional', '_diagnostic_text': 'finalize_4mu_final_09style_validation__diagnostic_text', '_finish_figure': 'finalize_4mu_final_09style_validation__finish_figure', '_scan_caption': 'finalize_4mu_final_09style_validation__scan_caption', '_final_caption': 'finalize_4mu_final_09style_validation__final_caption', '_plane_caption': 'finalize_4mu_final_09style_validation__plane_caption', '_comparison_specifications': 'finalize_4mu_final_09style_validation__comparison_specifications', '_diagnostic_overlay': 'finalize_4mu_final_09style_validation__diagnostic_overlay'})
_FINALIZE_NAMESPACES['abcd_4mu_final_data'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_4mu_final_data_Path', 'hashlib': 'finalize_4mu_final_data_hashlib', 'json': 'finalize_4mu_final_data_json', 'np': 'finalize_4mu_final_data_np', 'pd': 'finalize_4mu_final_data_pd', 'qr': 'finalize_4mu_final_data_qr', 'solve_triangular': 'finalize_4mu_final_data_solve_triangular', 'threadpool_limits': 'finalize_4mu_final_data_threadpool_limits', 'REGIONS': 'finalize_4mu_final_data_REGIONS', 'PUBLIC_SCOPES': 'finalize_4mu_final_data_PUBLIC_SCOPES', 'FINAL_A_SPECS': 'finalize_4mu_final_data_FINAL_A_SPECS', 'DATA_FIELDS': 'finalize_4mu_final_data_DATA_FIELDS', 'build_data_study': 'finalize_4mu_final_data_build_data_study', '_build': 'finalize_4mu_final_data__build', '_public_inputs': 'finalize_4mu_final_data__public_inputs', '_record': 'finalize_4mu_final_data__record', '_atoms': 'finalize_4mu_final_data__atoms', '_rectangle': 'finalize_4mu_final_data__rectangle', '_bounds': 'finalize_4mu_final_data__bounds', '_public_masks': 'finalize_4mu_final_data__public_masks', '_basis': 'finalize_4mu_final_data__basis', '_candidate_masks': 'finalize_4mu_final_data__candidate_masks', '_gates': 'finalize_4mu_final_data__gates', '_nominal_geometry': 'finalize_4mu_final_data__nominal_geometry', '_blank_statistics': 'finalize_4mu_final_data__blank_statistics', '_statistics': 'finalize_4mu_final_data__statistics', '_regress_selected': 'finalize_4mu_final_data__regress_selected', '_public_data_bins': 'finalize_4mu_final_data__public_data_bins', 'region': 'finalize_4mu_final_data_region'})
_FINALIZE_NAMESPACES['abcd_4mu_final_mc'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_4mu_final_mc_Path', 'ast': 'finalize_4mu_final_mc_ast', 'gc': 'finalize_4mu_final_mc_gc', 'hashlib': 'finalize_4mu_final_mc_hashlib', 'json': 'finalize_4mu_final_mc_json', 'pickle': 'finalize_4mu_final_mc_pickle', 'sys': 'finalize_4mu_final_mc_sys', 'np': 'finalize_4mu_final_mc_np', 'pd': 'finalize_4mu_final_mc_pd', 'SIDM_DIR': 'finalize_4mu_final_mc_SIDM_DIR', 'folder': 'finalize_4mu_final_mc_folder', 'common': 'finalize_4mu_final_mc_common', 'review': 'finalize_4mu_final_mc_review', 'statistics_from_regions': 'finalize_4mu_final_mc_statistics_from_regions', 'STATES': 'finalize_4mu_final_mc_STATES', 'REGIONS': 'finalize_4mu_final_mc_REGIONS', 'LOOSE_ISOLATION': 'finalize_4mu_final_mc_LOOSE_ISOLATION', 'LOOSE_ANGLES': 'finalize_4mu_final_mc_LOOSE_ANGLES', 'BUFFER_ISOLATION': 'finalize_4mu_final_mc_BUFFER_ISOLATION', 'BUFFER_ANGLES': 'finalize_4mu_final_mc_BUFFER_ANGLES', 'REFERENCE': 'finalize_4mu_final_mc_REFERENCE', 'CACHE_VERSION': 'finalize_4mu_final_mc_CACHE_VERSION', 'build_mc_study': 'finalize_4mu_final_mc_build_mc_study', 'candidate_specifications': 'finalize_4mu_final_mc_candidate_specifications', 'candidate_masks': 'finalize_4mu_final_mc_candidate_masks', 'candidate_regions': 'finalize_4mu_final_mc_candidate_regions', 'project_cube': 'finalize_4mu_final_mc_project_cube', '_tag': 'finalize_4mu_final_mc__tag', '_scopes': 'finalize_4mu_final_mc__scopes', '_load_mc': 'finalize_4mu_final_mc__load_mc', '_parameters': 'finalize_4mu_final_mc__parameters', '_finite_summary': 'finalize_4mu_final_mc__finite_summary', '_gate_summary': 'finalize_4mu_final_mc__gate_summary', '_scan_tables': 'finalize_4mu_final_mc__scan_tables', '_regression': 'finalize_4mu_final_mc__regression', '_projection_signature': 'finalize_4mu_final_mc__projection_signature'})
_FINALIZE_NAMESPACES['abcd_4mu_final_report'] = _FinalizeNamespace('abcd_finalize_common', {'datetime': 'finalize_4mu_final_report_datetime', 'timezone': 'finalize_4mu_final_report_timezone', 'hashlib': 'finalize_4mu_final_report_hashlib', 'json': 'finalize_4mu_final_report_json', 'Path': 'finalize_4mu_final_report_Path', 'plt': 'finalize_4mu_final_report_plt', 'PdfPages': 'finalize_4mu_final_report_PdfPages', 'np': 'finalize_4mu_final_report_np', 'pd': 'finalize_4mu_final_report_pd', 'SECTION_LABELS': 'finalize_4mu_final_report_SECTION_LABELS', 'build_4mu_final_report': 'finalize_4mu_final_report_build_4mu_final_report', 'file_hash': 'finalize_4mu_final_report_file_hash', 'protected_inputs': 'finalize_4mu_final_report_protected_inputs', 'diagnostic_candidates': 'finalize_4mu_final_report_diagnostic_candidates', 'summary_tables': 'finalize_4mu_final_report_summary_tables', 'wide_loose_audit': 'finalize_4mu_final_report_wide_loose_audit', 'figure_section': 'finalize_4mu_final_report_figure_section', 'json_default': 'finalize_4mu_final_report_json_default', 'save_report': 'finalize_4mu_final_report_save_report', 'show_4mu_section': 'finalize_4mu_final_report_show_4mu_section', '_export_numerical_tables': 'finalize_4mu_final_report__export_numerical_tables'})
_FINALIZE_NAMESPACES['abcd_buffer_distributions'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_buffer_distributions_Path', 'gc': 'finalize_buffer_distributions_gc', 'hashlib': 'finalize_buffer_distributions_hashlib', 'json': 'finalize_buffer_distributions_json', 'np': 'finalize_buffer_distributions_np', 'pd': 'finalize_buffer_distributions_pd', 'common': 'finalize_buffer_distributions_common', 'scan_validation': 'finalize_buffer_distributions_scan_validation', 'sideband_validation': 'finalize_buffer_distributions_sideband_validation', 'candidate_coordinates': 'finalize_buffer_distributions_candidate_coordinates', '_opt_distribution_edges': 'finalize_buffer_distributions__opt_distribution_edges', '_opt_fold_distribution': 'finalize_buffer_distributions__opt_fold_distribution', 'strata_specs': 'finalize_buffer_distributions_strata_specs', 'project_strata': 'finalize_buffer_distributions_project_strata', 'project_mass_partitions': 'finalize_buffer_distributions_project_mass_partitions', '_state_index': 'finalize_buffer_distributions__state_index', 'BUFFER': 'finalize_buffer_distributions_BUFFER', 'SCOPES': 'finalize_buffer_distributions_SCOPES', 'ALLOWED_DATA_SCOPES': 'finalize_buffer_distributions_ALLOWED_DATA_SCOPES', 'FLOW_NOTE': 'finalize_buffer_distributions_FLOW_NOTE', 'run_buffer_distributions': 'finalize_buffer_distributions_run_buffer_distributions', '_audit_latest_snapshot': 'finalize_buffer_distributions__audit_latest_snapshot', '_display_priors': 'finalize_buffer_distributions__display_priors', '_physical_bins': 'finalize_buffer_distributions__physical_bins', '_native_atomic': 'finalize_buffer_distributions__native_atomic', '_bin_mask': 'finalize_buffer_distributions__bin_mask', '_regress_buffer_mc': 'finalize_buffer_distributions__regress_buffer_mc', '_dd_mc_plane': 'finalize_buffer_distributions__dd_mc_plane', 'load_buffer_mc': 'finalize_buffer_distributions_load_buffer_mc', '_load_data_bins': 'finalize_buffer_distributions__load_data_bins', '_validate_visible_counts': 'finalize_buffer_distributions__validate_visible_counts', '_regress_nominal_mc': 'finalize_buffer_distributions__regress_nominal_mc'})
_FINALIZE_NAMESPACES['abcd_capped_gap_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'np': 'finalize_capped_gap_plots_np', 'pd': 'finalize_capped_gap_plots_pd', 'plt': 'finalize_capped_gap_plots_plt', 'Line2D': 'finalize_capped_gap_plots_Line2D', '_canvas': 'finalize_capped_gap_plots__canvas', '_caption': 'finalize_capped_gap_plots__caption', '_style_table': 'finalize_capped_gap_plots__style_table', 'capped_gap_figures': 'finalize_capped_gap_plots_capped_gap_figures', 'capped_gap_definitions': 'finalize_capped_gap_plots_capped_gap_definitions', '_gap_tradeoff': 'finalize_capped_gap_plots__gap_tradeoff', '_gap_support': 'finalize_capped_gap_plots__gap_support', '_gap_selection': 'finalize_capped_gap_plots__gap_selection'})
_FINALIZE_NAMESPACES['abcd_comprehensive'] = _FinalizeNamespace('abcd_finalize_common', {'datetime': 'finalize_comprehensive_datetime', 'timezone': 'finalize_comprehensive_timezone', 'Path': 'finalize_comprehensive_Path', 'json': 'finalize_comprehensive_json', 'np': 'finalize_comprehensive_np', 'pd': 'finalize_comprehensive_pd', 'PdfPages': 'finalize_comprehensive_PdfPages', 'common': 'finalize_comprehensive_common', 'review': 'finalize_comprehensive_review', 'run_mass_inclusive': 'finalize_comprehensive_run_mass_inclusive', 'REFERENCE': 'finalize_comprehensive_REFERENCE', 'ORDER': 'finalize_comprehensive_ORDER', 'comprehensive_points': 'finalize_comprehensive_comprehensive_points', '_read': 'finalize_comprehensive__read', '_at': 'finalize_comprehensive__at', 'contamination_diagnostics': 'finalize_comprehensive_contamination_diagnostics', '_geometry': 'finalize_comprehensive__geometry', 'nominal_diagnostics': 'finalize_comprehensive_nominal_diagnostics', 'run_comprehensive': 'finalize_comprehensive_run_comprehensive', 'export_comprehensive': 'finalize_comprehensive_export_comprehensive'})
_FINALIZE_NAMESPACES['abcd_comprehensive_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'textwrap': 'finalize_comprehensive_plots_textwrap', 'plt': 'finalize_comprehensive_plots_plt', 'LogNorm': 'finalize_comprehensive_plots_LogNorm', 'Line2D': 'finalize_comprehensive_plots_Line2D', 'Patch': 'finalize_comprehensive_plots_Patch', 'Rectangle': 'finalize_comprehensive_plots_Rectangle', 'np': 'finalize_comprehensive_plots_np', 'pd': 'finalize_comprehensive_plots_pd', 'reference_plots': 'finalize_comprehensive_plots_reference_plots', 'local_plots': 'finalize_comprehensive_plots_local_plots', 'PLANES': 'finalize_comprehensive_plots_PLANES', 'REGION_COLORS': 'finalize_comprehensive_plots_REGION_COLORS', 'MASS_COLORS': 'finalize_comprehensive_plots_MASS_COLORS', 'MASS_LABELS': 'finalize_comprehensive_plots_MASS_LABELS', '_table': 'finalize_comprehensive_plots__table', '_points': 'finalize_comprehensive_plots__points', '_labels': 'finalize_comprehensive_plots__labels', '_canvas': 'finalize_comprehensive_plots__canvas', '_caption': 'finalize_comprehensive_plots__caption', '_cut_rows': 'finalize_comprehensive_plots__cut_rows', '_positive_line': 'finalize_comprehensive_plots__positive_line', '_closure': 'finalize_comprehensive_plots__closure', '_sb_regions': 'finalize_comprehensive_plots__sb_regions', '_sb_decomposition': 'finalize_comprehensive_plots__sb_decomposition', '_nominal_retention': 'finalize_comprehensive_plots__nominal_retention', '_nominal_closure': 'finalize_comprehensive_plots__nominal_closure', '_nominal_purity': 'finalize_comprehensive_plots__nominal_purity', '_blind_overlap': 'finalize_comprehensive_plots__blind_overlap', '_data_access': 'finalize_comprehensive_plots__data_access', '_mass_rows': 'finalize_comprehensive_plots__mass_rows', '_intermediate_rows': 'finalize_comprehensive_plots__intermediate_rows', '_mass_closure': 'finalize_comprehensive_plots__mass_closure', '_mass_components': 'finalize_comprehensive_plots__mass_components', '_mass_purity': 'finalize_comprehensive_plots__mass_purity', '_mass_distribution': 'finalize_comprehensive_plots__mass_distribution', '_mass_scan': 'finalize_comprehensive_plots__mass_scan', 'comprehensive_figure_definitions': 'finalize_comprehensive_plots_comprehensive_figure_definitions', 'make_comprehensive_figures': 'finalize_comprehensive_plots_make_comprehensive_figures'})
_FINALIZE_NAMESPACES['abcd_data_expected_significance'] = _FinalizeNamespace('abcd_finalize_plots', {'np': 'finalize_data_expected_significance_np', 'pd': 'finalize_data_expected_significance_pd', 'Line2D': 'finalize_data_expected_significance_Line2D', '_canvas': 'finalize_data_expected_significance__canvas', '_caption': 'finalize_data_expected_significance__caption', 'COLORS': 'finalize_data_expected_significance_COLORS', 'LABELS': 'finalize_data_expected_significance_LABELS', 'final_comparison_tables': 'finalize_data_expected_significance_final_comparison_tables', 'asimov_significance': 'finalize_data_expected_significance_asimov_significance', 'METRICS': 'finalize_data_expected_significance_METRICS', 'data_expected_significance_definitions': 'finalize_data_expected_significance_data_expected_significance_definitions', 'data_expected_significance_figures': 'finalize_data_expected_significance_data_expected_significance_figures', 'data_expected_significance_tables': 'finalize_data_expected_significance_data_expected_significance_tables', '_quadrature_expected_z': 'finalize_data_expected_significance__quadrature_expected_z', '_data_expected_plot': 'finalize_data_expected_significance__data_expected_plot'})
_FINALIZE_NAMESPACES['abcd_final_check'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_final_check_Path', 'gc': 'finalize_final_check_gc', 'json': 'finalize_final_check_json', 'np': 'finalize_final_check_np', 'pd': 'finalize_final_check_pd', 'common': 'finalize_final_check_common', 'build_vr2_focus_config': 'finalize_final_check_build_vr2_focus_config', 'project_mass_partitions': 'finalize_final_check_project_mass_partitions', 'POINTS': 'finalize_final_check_POINTS', 'STATES': 'finalize_final_check_STATES', 'MASS_SLICES': 'finalize_final_check_MASS_SLICES', 'NUMERIC_STATS': 'finalize_final_check_NUMERIC_STATS', 'build_final_check_config': 'finalize_final_check_build_final_check_config', '_read': 'finalize_final_check__read', '_selected': 'finalize_final_check__selected', '_at': 'finalize_final_check__at', '_metrics': 'finalize_final_check__metrics', '_bounds': 'finalize_final_check__bounds', '_masks': 'finalize_final_check__masks', '_load_backgrounds': 'finalize_final_check__load_backgrounds', '_nominal_tables': 'finalize_final_check__nominal_tables', '_migration_tables': 'finalize_final_check__migration_tables', '_cache_tables': 'finalize_final_check__cache_tables', '_mass_tables': 'finalize_final_check__mass_tables', 'run_final_check': 'finalize_final_check_run_final_check', 'r': 'finalize_final_check_r'})
_FINALIZE_NAMESPACES['abcd_final_check_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'textwrap': 'finalize_final_check_plots_textwrap', 'plt': 'finalize_final_check_plots_plt', 'Line2D': 'finalize_final_check_plots_Line2D', 'Patch': 'finalize_final_check_plots_Patch', 'np': 'finalize_final_check_plots_np', 'pd': 'finalize_final_check_plots_pd', 'reference_plots': 'finalize_final_check_plots_reference_plots', 'COMPONENT_COLORS': 'finalize_final_check_plots_COMPONENT_COLORS', 'WPS': 'finalize_final_check_plots_WPS', 'WP_COLORS': 'finalize_final_check_plots_WP_COLORS', 'PLANES': 'finalize_final_check_plots_PLANES', 'MASSES': 'finalize_final_check_plots_MASSES', 'MASS_LABELS': 'finalize_final_check_plots_MASS_LABELS', 'MASS_COLORS': 'finalize_final_check_plots_MASS_COLORS', 'STEPS': 'finalize_final_check_plots_STEPS', 'STEP_LABELS': 'finalize_final_check_plots_STEP_LABELS', '_table': 'finalize_final_check_plots__table', '_canvas': 'finalize_final_check_plots__canvas', '_caption': 'finalize_final_check_plots__caption', '_wp_labels': 'finalize_final_check_plots__wp_labels', '_format': 'finalize_final_check_plots__format', '_style_table': 'finalize_final_check_plots__style_table', '_summary': 'finalize_final_check_plots__summary', '_nominal_support': 'finalize_final_check_plots__nominal_support', '_sample_leverage': 'finalize_final_check_plots__sample_leverage', '_migration': 'finalize_final_check_plots__migration', '_nominal_distribution': 'finalize_final_check_plots__nominal_distribution', '_mass_closure': 'finalize_final_check_plots__mass_closure', '_mass_composition': 'finalize_final_check_plots__mass_composition', '_mass_mixing': 'finalize_final_check_plots__mass_mixing', '_c_contamination': 'finalize_final_check_plots__c_contamination', '_significance': 'finalize_final_check_plots__significance', '_paired': 'finalize_final_check_plots__paired', '_safety': 'finalize_final_check_plots__safety', 'final_check_figure_definitions': 'finalize_final_check_plots_final_check_figure_definitions', 'make_final_check_figures': 'finalize_final_check_plots_make_final_check_figures'})
_FINALIZE_NAMESPACES['abcd_final_check_strata'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_final_check_strata_Path', 'gc': 'finalize_final_check_strata_gc', 'np': 'finalize_final_check_strata_np', 'pd': 'finalize_final_check_strata_pd', 'common': 'finalize_final_check_strata_common', 'POINTS': 'finalize_final_check_strata_POINTS', 'NUMERIC_STATS': 'finalize_final_check_strata_NUMERIC_STATS', '_state_index': 'finalize_final_check_strata__state_index', 'FINE_MASSES': 'finalize_final_check_strata_FINE_MASSES', 'COARSE_MASSES': 'finalize_final_check_strata_COARSE_MASSES', 'STATES': 'finalize_final_check_strata_STATES', 'RATIO_FIELDS': 'finalize_final_check_strata_RATIO_FIELDS', 'DATA_NUMERIC_STATS': 'finalize_final_check_strata_DATA_NUMERIC_STATS', '_settings': 'finalize_final_check_strata__settings', '_specs': 'finalize_final_check_strata__specs', '_sum_planes': 'finalize_final_check_strata__sum_planes', '_selection_index': 'finalize_final_check_strata__selection_index', '_project_mc': 'finalize_final_check_strata__project_mc', '_load_mc': 'finalize_final_check_strata__load_mc', '_parent_plane': 'finalize_final_check_strata__parent_plane', '_validate_parents': 'finalize_final_check_strata__validate_parents', '_ratio': 'finalize_final_check_strata__ratio', '_statistics': 'finalize_final_check_strata__statistics', '_purity_tables': 'finalize_final_check_strata__purity_tables', '_closure_table': 'finalize_final_check_strata__closure_table', '_composition_table': 'finalize_final_check_strata__composition_table', '_validate_output': 'finalize_final_check_strata__validate_output', 'run_stratified_checks': 'finalize_final_check_strata_run_stratified_checks', '_project_data_leaf': 'finalize_final_check_strata__project_data_leaf', '_load_approved_data': 'finalize_final_check_strata__load_approved_data', '_validate_data_parents': 'finalize_final_check_strata__validate_data_parents', 'finalize_stratified_data': 'finalize_final_check_strata_finalize_stratified_data'})
_FINALIZE_NAMESPACES['abcd_final_comparison_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'re': 'finalize_final_comparison_plots_re', 'np': 'finalize_final_comparison_plots_np', 'pd': 'finalize_final_comparison_plots_pd', 'Line2D': 'finalize_final_comparison_plots_Line2D', 'Patch': 'finalize_final_comparison_plots_Patch', '_canvas': 'finalize_final_comparison_plots__canvas', '_caption': 'finalize_final_comparison_plots__caption', '_closure_axis': 'finalize_final_comparison_plots__closure_axis', '_closure_points': 'finalize_final_comparison_plots__closure_points', 'COLORS': 'finalize_final_comparison_plots_COLORS', 'LABELS': 'finalize_final_comparison_plots_LABELS', 'METRICS': 'finalize_final_comparison_plots_METRICS', 'final_comparison_definitions': 'finalize_final_comparison_plots_final_comparison_definitions', 'make_final_comparison_figures': 'finalize_final_comparison_plots_make_final_comparison_figures', 'final_comparison_tables': 'finalize_final_comparison_plots_final_comparison_tables', '_intermediate_overlay': 'finalize_final_comparison_plots__intermediate_overlay', '_significance_comparison': 'finalize_final_comparison_plots__significance_comparison'})
_FINALIZE_NAMESPACES['abcd_final_nominal_controls'] = _FinalizeNamespace('abcd_finalize_common', {'np': 'finalize_final_nominal_controls_np', 'pd': 'finalize_final_nominal_controls_pd', 'IDENTITY': 'finalize_final_nominal_controls_IDENTITY', '_load_approved_sidebands': 'finalize_final_nominal_controls__load_approved_sidebands', 'data_prediction': 'finalize_final_nominal_controls_data_prediction', '_overlap': 'finalize_final_nominal_controls__overlap', 'region_rectangles': 'finalize_final_nominal_controls_region_rectangles', 'evaluate_final_nominal_controls': 'finalize_final_nominal_controls_evaluate_final_nominal_controls'})
_FINALIZE_NAMESPACES['abcd_finalize_common'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'Path', 'gc': 'gc', 'hashlib': 'hashlib', 'json': 'json', 'sys': 'sys', 'datetime': 'datetime', 'timezone': 'timezone', 'coffea': 'coffea', 'np': 'np', 'pd': 'pd', '_numpy_core': '_numpy_core', '_numpy_multiarray': '_numpy_multiarray', '_numpy_numeric': '_numpy_numeric', 'COMPARISON_VARIABLES': 'COMPARISON_VARIABLES', '_comparison_json': '_comparison_json', '_comparison_hash': '_comparison_hash', '_comparison_group': '_comparison_group', '_comparison_split': '_comparison_split', '_comparison_validate_arrays': '_comparison_validate_arrays', '_comparison_plane': '_comparison_plane', '_comparison_add': '_comparison_add', '_comparison_payload': '_comparison_payload', '_comparison_validate_hist': '_comparison_validate_hist', '_comparison_current_cube': '_comparison_current_cube', 'plane_statistics': 'plane_statistics', '_extension_coarsen_regular': '_extension_coarsen_regular', 'extension_correlations': 'extension_correlations', '_extension_data_payload': '_extension_data_payload', '_extension_blank_statistics': '_extension_blank_statistics', '_angle_regions': '_angle_regions', '_comparison_regions': '_comparison_regions', 'candidate_specs': 'candidate_specs', 'candidate_indices': 'candidate_indices', 'project_candidate': 'project_candidate', 'add_candidate': 'add_candidate', 'candidate_input_record': 'candidate_input_record', 'candidate_signal_paths': 'candidate_signal_paths', 'load_candidate_mc': 'load_candidate_mc', 'candidate_coordinates': 'candidate_coordinates', 'candidate_purity': 'candidate_purity', 'load_candidate_data': 'load_candidate_data', 'collect_candidate_tables': 'collect_candidate_tables', 'validate_candidate_tables': 'validate_candidate_tables', 'candidate_summary': 'candidate_summary', 'run_candidate_comparison': 'run_candidate_comparison', 'export_candidate_comparison': 'export_candidate_comparison', 'mc_sr_spec': 'mc_sr_spec', 'project_mc_sr': 'project_mc_sr', 'collect_mc_sr_reference': 'collect_mc_sr_reference', 'validate_mc_sr_reference': 'validate_mc_sr_reference', 'validate_finalize_config': 'validate_finalize_config', 'validate_vr_spec': 'validate_vr_spec', 'mc_sr_full_grid_path': 'mc_sr_full_grid_path', 'validate_mc_sr_partition': 'validate_mc_sr_partition', 'candidate_distribution_purity': 'candidate_distribution_purity'})
_FINALIZE_NAMESPACES['abcd_finalize_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'textwrap': 'textwrap', 'np': 'np', 'pd': 'pd', 'plt': 'plt', 'LogNorm': 'LogNorm', 'Normalize': 'Normalize', 'SymLogNorm': 'SymLogNorm', 'TwoSlopeNorm': 'TwoSlopeNorm', 'Line2D': 'Line2D', 'Patch': 'Patch', 'Rectangle': 'Rectangle', '_opt_ORDER': '_opt_ORDER', '_opt_SAMPLES': '_opt_SAMPLES', '_opt_COLORS': '_opt_COLORS', '_opt_frame': '_opt_frame', '_opt_wp': '_opt_wp', '_opt_angular': '_opt_angular', '_opt_title': '_opt_title', '_opt_channel': '_opt_channel', '_opt_figure': '_opt_figure', '_opt_caption': '_opt_caption', '_opt_unavailable': '_opt_unavailable', '_opt_norm': '_opt_norm', '_opt_errorbar': '_opt_errorbar', '_opt_dynamic_scale': '_opt_dynamic_scale', '_opt_distribution_figures': '_opt_distribution_figures', '_opt_closure_figure': '_opt_closure_figure', '_opt_cell_edges': '_opt_cell_edges', '_opt_grid_figures': '_opt_grid_figures', '_opt_composition_figures': '_opt_composition_figures', '_opt_correlation_figure': '_opt_correlation_figure', '_opt_closure_limits': '_opt_closure_limits', '_opt_separate_y_ticks': '_opt_separate_y_ticks', '_opt_valid_error': '_opt_valid_error', '_opt_reference_errorbar': '_opt_reference_errorbar', '_opt_data_missing_markers': '_opt_data_missing_markers', '_opt_summary_number': '_opt_summary_number', '_opt_summary_finish': '_opt_summary_finish', '_opt_distribution_edges': '_opt_distribution_edges', '_opt_fold_distribution': '_opt_fold_distribution', '_opt_distribution_norm': '_opt_distribution_norm', '_candidate_purity_limits': '_candidate_purity_limits', '_candidate_purity_axis': '_candidate_purity_axis', '_candidate_scan_diagnostics': '_candidate_scan_diagnostics', '_opt_scan_figures': '_opt_scan_figures', '_candidate_purity_figure': '_candidate_purity_figure', '_candidate_combined_closure': '_candidate_combined_closure', 'make_comparison_figures': 'make_comparison_figures', 'make_sr_reference_figures': 'make_sr_reference_figures', '_opt_data_approved': '_opt_data_approved', '_opt_mask_distribution': '_opt_mask_distribution'})
_FINALIZE_NAMESPACES['abcd_inclusive_dp_bins'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_inclusive_dp_bins_Path', 'monotonic': 'finalize_inclusive_dp_bins_monotonic', 'np': 'finalize_inclusive_dp_bins_np', 'pd': 'finalize_inclusive_dp_bins_pd', 'buffered': 'finalize_inclusive_dp_bins_buffered', 'common': 'finalize_inclusive_dp_bins_common', 'scan_validation': 'finalize_inclusive_dp_bins_scan_validation', 'audit_inclusive_dp_bins': 'finalize_inclusive_dp_bins_audit_inclusive_dp_bins', '_validate_original_mc': 'finalize_inclusive_dp_bins__validate_original_mc', '_validated_display_priors': 'finalize_inclusive_dp_bins__validated_display_priors', '_validate_display_mc': 'finalize_inclusive_dp_bins__validate_display_mc', '_add_public_integrals': 'finalize_inclusive_dp_bins__add_public_integrals', '_validate_grid_mc': 'finalize_inclusive_dp_bins__validate_grid_mc'})
_FINALIZE_NAMESPACES['abcd_iso_buffer_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'np': 'finalize_iso_buffer_plots_np', 'pd': 'finalize_iso_buffer_plots_pd', 'plt': 'finalize_iso_buffer_plots_plt', 'patheffects': 'finalize_iso_buffer_plots_patheffects', 'LogNorm': 'finalize_iso_buffer_plots_LogNorm', 'Normalize': 'finalize_iso_buffer_plots_Normalize', 'Line2D': 'finalize_iso_buffer_plots_Line2D', 'Patch': 'finalize_iso_buffer_plots_Patch', 'Rectangle': 'finalize_iso_buffer_plots_Rectangle', '_canvas': 'finalize_iso_buffer_plots__canvas', '_caption': 'finalize_iso_buffer_plots__caption', '_style_table': 'finalize_iso_buffer_plots__style_table', '_data_mask': 'finalize_iso_buffer_plots__data_mask', 'ANGLES': 'finalize_iso_buffer_plots_ANGLES', 'COLORS': 'finalize_iso_buffer_plots_COLORS', 'MASS_TITLES': 'finalize_iso_buffer_plots_MASS_TITLES', 'SCOPE_PREFIX': 'finalize_iso_buffer_plots_SCOPE_PREFIX', '_scan': 'finalize_iso_buffer_plots__scan', '_best': 'finalize_iso_buffer_plots__best', '_final': 'finalize_iso_buffer_plots__final', '_iso_axis': 'finalize_iso_buffer_plots__iso_axis', '_closure_axis': 'finalize_iso_buffer_plots__closure_axis', '_closure_points': 'finalize_iso_buffer_plots__closure_points', '_tradeoff': 'finalize_iso_buffer_plots__tradeoff', '_nominal_closure': 'finalize_iso_buffer_plots__nominal_closure', '_selection': 'finalize_iso_buffer_plots__selection', '_vr_scan': 'finalize_iso_buffer_plots__vr_scan', '_selected_note': 'finalize_iso_buffer_plots__selected_note', '_gap_nominal_closure': 'finalize_iso_buffer_plots__gap_nominal_closure', '_path_frame': 'finalize_iso_buffer_plots__path_frame', '_gap_vr_scan': 'finalize_iso_buffer_plots__gap_vr_scan', '_final_1d': 'finalize_iso_buffer_plots__final_1d', '_planes': 'finalize_iso_buffer_plots__planes', '_geometry': 'finalize_iso_buffer_plots__geometry', '_cut': 'finalize_iso_buffer_plots__cut', '_overlay': 'finalize_iso_buffer_plots__overlay', '_plane_image': 'finalize_iso_buffer_plots__plane_image', '_distribution_legend': 'finalize_iso_buffer_plots__distribution_legend', '_nominal_plane': 'finalize_iso_buffer_plots__nominal_plane', '_nominal_distribution': 'finalize_iso_buffer_plots__nominal_distribution', '_vr_distributions': 'finalize_iso_buffer_plots__vr_distributions', '_reason': 'finalize_iso_buffer_plots__reason', '_norm': 'finalize_iso_buffer_plots__norm', '_nominal_fraction_table': 'finalize_iso_buffer_plots__nominal_fraction_table', '_definition_label': 'finalize_iso_buffer_plots__definition_label', '_nominal_data_contamination': 'finalize_iso_buffer_plots__nominal_data_contamination', '_approaches': 'finalize_iso_buffer_plots__approaches', '_meeting_comparison': 'finalize_iso_buffer_plots__meeting_comparison', 'iso_buffer_figure_definitions': 'finalize_iso_buffer_plots_iso_buffer_figure_definitions', 'make_iso_buffer_figures': 'finalize_iso_buffer_plots_make_iso_buffer_figures', '_visible_states': 'finalize_iso_buffer_plots__visible_states'})
_FINALIZE_NAMESPACES['abcd_iso_focus'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_iso_focus_Path', 'hashlib': 'finalize_iso_focus_hashlib', 'json': 'finalize_iso_focus_json', 'np': 'finalize_iso_focus_np', 'pd': 'finalize_iso_focus_pd', 'common': 'finalize_iso_focus_common', 'review': 'finalize_iso_focus_review', 'PROTECTED': 'finalize_iso_focus_PROTECTED', '_coordinate_name': 'finalize_iso_focus__coordinate_name', '_overlap': 'finalize_iso_focus__overlap', '_quantiles': 'finalize_iso_focus__quantiles', 'region_rectangles': 'finalize_iso_focus_region_rectangles', 'statistics_from_regions': 'finalize_iso_focus_statistics_from_regions', 'MEASUREMENT_FIELDS': 'finalize_iso_focus_MEASUREMENT_FIELDS', '_load_caches': 'finalize_iso_focus__load_caches', 'ANGLES': 'finalize_iso_focus_ANGLES', 'REGIONS': 'finalize_iso_focus_REGIONS', 'SCOPES': 'finalize_iso_focus_SCOPES', 'build_iso_focus': 'finalize_iso_focus_build_iso_focus', '_read': 'finalize_iso_focus__read', '_validate_previous': 'finalize_iso_focus__validate_previous', '_select': 'finalize_iso_focus__select', '_vr_tables': 'finalize_iso_focus__vr_tables', '_nominal_data_contamination': 'finalize_iso_focus__nominal_data_contamination', '_published_nominal_match': 'finalize_iso_focus__published_nominal_match', '_validate_public_vr_grids': 'finalize_iso_focus__validate_public_vr_grids', 'attach_final_nominal_controls': 'finalize_iso_focus_attach_final_nominal_controls'})
_FINALIZE_NAMESPACES['abcd_local_scan'] = _FinalizeNamespace('abcd_finalize_common', {'datetime': 'finalize_local_scan_datetime', 'timezone': 'finalize_local_scan_timezone', 'gc': 'finalize_local_scan_gc', 'json': 'finalize_local_scan_json', 'Path': 'finalize_local_scan_Path', 'np': 'finalize_local_scan_np', 'pd': 'finalize_local_scan_pd', 'common': 'finalize_local_scan_common', 'review': 'finalize_local_scan_review', 'LOCAL_ISO': 'finalize_local_scan_LOCAL_ISO', 'LOCAL_DPHI': 'finalize_local_scan_LOCAL_DPHI', 'REFERENCE_WP': 'finalize_local_scan_REFERENCE_WP', 'SCOPE_ORDER': 'finalize_local_scan_SCOPE_ORDER', 'REGIONS': 'finalize_local_scan_REGIONS', 'STATISTICS': 'finalize_local_scan_STATISTICS', 'build_config': 'finalize_local_scan_build_config', '_vr_config': 'finalize_local_scan__vr_config', '_check_local_config': 'finalize_local_scan__check_local_config', '_coordinates': 'finalize_local_scan__coordinates', '_read_table': 'finalize_local_scan__read_table', '_local_rows': 'finalize_local_scan__local_rows', '_validate_cache_manifest': 'finalize_local_scan__validate_cache_manifest', 'load_cached_tables': 'finalize_local_scan_load_cached_tables', 'expected_scopes': 'finalize_local_scan_expected_scopes', '_load_background_planes': 'finalize_local_scan__load_background_planes', '_validate_projected_planes': 'finalize_local_scan__validate_projected_planes', '_composition_tables': 'finalize_local_scan__composition_tables', '_composition_transfer': 'finalize_local_scan__composition_transfer', 'run_local_scan': 'finalize_local_scan_run_local_scan', 'export_local_scan': 'finalize_local_scan_export_local_scan', 'build_vr2_focus_config': 'finalize_local_scan_build_vr2_focus_config', 'run_vr2_focus': 'finalize_local_scan_run_vr2_focus', 'vr2_focus_summary_tables': 'finalize_local_scan_vr2_focus_summary_tables', 'region': 'finalize_local_scan_region'})
_FINALIZE_NAMESPACES['abcd_local_scan_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'textwrap': 'finalize_local_scan_plots_textwrap', 'plt': 'finalize_local_scan_plots_plt', 'LogNorm': 'finalize_local_scan_plots_LogNorm', 'Normalize': 'finalize_local_scan_plots_Normalize', 'TwoSlopeNorm': 'finalize_local_scan_plots_TwoSlopeNorm', 'Line2D': 'finalize_local_scan_plots_Line2D', 'Rectangle': 'finalize_local_scan_plots_Rectangle', 'ScalarFormatter': 'finalize_local_scan_plots_ScalarFormatter', 'np': 'finalize_local_scan_plots_np', 'pd': 'finalize_local_scan_plots_pd', 'SCOPES': 'finalize_local_scan_plots_SCOPES', 'COLORS': 'finalize_local_scan_plots_COLORS', 'COMPONENT_COLORS': 'finalize_local_scan_plots_COMPONENT_COLORS', '_frame': 'finalize_local_scan_plots__frame', '_scope_name': 'finalize_local_scan_plots__scope_name', '_points': 'finalize_local_scan_plots__points', '_caption': 'finalize_local_scan_plots__caption', '_canvas': 'finalize_local_scan_plots__canvas', '_number': 'finalize_local_scan_plots__number', '_grid': 'finalize_local_scan_plots__grid', '_draw_grid': 'finalize_local_scan_plots__draw_grid', '_map_page': 'finalize_local_scan_plots__map_page', '_overview': 'finalize_local_scan_plots__overview', '_ranking_table': 'finalize_local_scan_plots__ranking_table', '_sr_maps': 'finalize_local_scan_plots__sr_maps', '_signal_pages': 'finalize_local_scan_plots__signal_pages', '_forest': 'finalize_local_scan_plots__forest', '_yield_page': 'finalize_local_scan_plots__yield_page', '_support_page': 'finalize_local_scan_plots__support_page', '_contamination': 'finalize_local_scan_plots__contamination', '_contamination_regions': 'finalize_local_scan_plots__contamination_regions', '_composition_page': 'finalize_local_scan_plots__composition_page', '_transfer_page': 'finalize_local_scan_plots__transfer_page', '_dominance_page': 'finalize_local_scan_plots__dominance_page', '_decision_page': 'finalize_local_scan_plots__decision_page', 'local_scan_figure_definitions': 'finalize_local_scan_plots_local_scan_figure_definitions', 'make_local_scan_figures': 'finalize_local_scan_plots_make_local_scan_figures', 'common': 'finalize_local_scan_plots_common', '_scan_selected': 'finalize_local_scan_plots__scan_selected', '_scan_wp_label': 'finalize_local_scan_plots__scan_wp_label', '_scan_point': 'finalize_local_scan_plots__scan_point', '_scan_scope_text': 'finalize_local_scan_plots__scan_scope_text', '_scan_series': 'finalize_local_scan_plots__scan_series', '_local_scan_figure': 'finalize_local_scan_plots__local_scan_figure', '_scan_conditional_distribution': 'finalize_local_scan_plots__scan_conditional_distribution', '_local_distributions_figure': 'finalize_local_scan_plots__local_distributions_figure', 'Patch': 'finalize_local_scan_plots_Patch', '_statplot_points': 'finalize_local_scan_plots__statplot_points', '_statplot_labels': 'finalize_local_scan_plots__statplot_labels', '_statplot_ticks': 'finalize_local_scan_plots__statplot_ticks', '_statplot_highlight': 'finalize_local_scan_plots__statplot_highlight', '_statplot_note': 'finalize_local_scan_plots__statplot_note', '_local_toy_rankings': 'finalize_local_scan_plots__local_toy_rankings', '_local_toy_covariance': 'finalize_local_scan_plots__local_toy_covariance', '_local_likelihood_figure': 'finalize_local_scan_plots__local_likelihood_figure', '_vr2focus_points': 'finalize_local_scan_plots__vr2focus_points', '_vr2focus_rows': 'finalize_local_scan_plots__vr2focus_rows', '_vr2focus_edges': 'finalize_local_scan_plots__vr2focus_edges', '_vr2focus_legend': 'finalize_local_scan_plots__vr2focus_legend', '_vr2focus_full_map': 'finalize_local_scan_plots__vr2focus_full_map', '_vr2focus_joint': 'finalize_local_scan_plots__vr2focus_joint', '_vr2focus_summary': 'finalize_local_scan_plots__vr2focus_summary', 'vr2_focus_figure_definitions': 'finalize_local_scan_plots_vr2_focus_figure_definitions', 'make_vr2_focus_figures': 'finalize_local_scan_plots_make_vr2_focus_figures', '_vr2focus_significance': 'finalize_local_scan_plots__vr2focus_significance'})
_FINALIZE_NAMESPACES['abcd_local_scan_statistics'] = _FinalizeNamespace('abcd_finalize_common', {'np': 'finalize_local_scan_statistics_np', 'pd': 'finalize_local_scan_statistics_pd', 'brentq': 'finalize_local_scan_statistics_brentq', 'minimize': 'finalize_local_scan_statistics_minimize', 'rankdata': 'finalize_local_scan_statistics_rankdata', 'asimov_significance': 'finalize_local_scan_statistics_asimov_significance', '_coordinate_values': 'finalize_local_scan_statistics__coordinate_values', '_local_partition': 'finalize_local_scan_statistics__local_partition', '_cumulative_to_atoms': 'finalize_local_scan_statistics__cumulative_to_atoms', '_positive_moments': 'finalize_local_scan_statistics__positive_moments', '_moment_draws': 'finalize_local_scan_statistics__moment_draws', '_rank_frequencies': 'finalize_local_scan_statistics__rank_frequencies', 'local_rank_diagnostics': 'finalize_local_scan_statistics_local_rank_diagnostics', 'asimov_abcd_significance': 'finalize_local_scan_statistics_asimov_abcd_significance', 'local_likelihood_scenarios': 'finalize_local_scan_statistics_local_likelihood_scenarios', 'validate_stats': 'finalize_local_scan_statistics_validate_stats', 'add_local_statistics': 'finalize_local_scan_statistics_add_local_statistics', 'local_likelihood_inputs': 'finalize_local_scan_statistics_local_likelihood_inputs'})
_FINALIZE_NAMESPACES['abcd_mass_inclusive'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_mass_inclusive_Path', 'gc': 'finalize_mass_inclusive_gc', 'np': 'finalize_mass_inclusive_np', 'pd': 'finalize_mass_inclusive_pd', 'common': 'finalize_mass_inclusive_common', 'STATES': 'finalize_mass_inclusive_STATES', 'MASS_SLICES': 'finalize_mass_inclusive_MASS_SLICES', '_settings': 'finalize_mass_inclusive__settings', '_specs': 'finalize_mass_inclusive__specs', '_state_index': 'finalize_mass_inclusive__state_index', 'project_mass_partitions': 'finalize_mass_inclusive_project_mass_partitions', '_project_all_data': 'finalize_mass_inclusive__project_all_data', '_load_mc': 'finalize_mass_inclusive__load_mc', '_validate_middle_mass': 'finalize_mass_inclusive__validate_middle_mass', '_mass_component_table': 'finalize_mass_inclusive__mass_component_table', '_load_data': 'finalize_mass_inclusive__load_data', '_validate_tables': 'finalize_mass_inclusive__validate_tables', 'run_mass_inclusive': 'finalize_mass_inclusive_run_mass_inclusive'})
_FINALIZE_NAMESPACES['abcd_selected_vr_displays'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_selected_vr_displays_Path', 'copy': 'finalize_selected_vr_displays_copy', 'gc': 'finalize_selected_vr_displays_gc', 'json': 'finalize_selected_vr_displays_json', 'np': 'finalize_selected_vr_displays_np', 'pd': 'finalize_selected_vr_displays_pd', 'buffered': 'finalize_selected_vr_displays_buffered', 'common': 'finalize_selected_vr_displays_common', 'build_iso_focus': 'finalize_selected_vr_displays_build_iso_focus', 'candidate_coordinates': 'finalize_selected_vr_displays_candidate_coordinates', 'SELECTIONS': 'finalize_selected_vr_displays_SELECTIONS', 'run_selected_vr_displays': 'finalize_selected_vr_displays_run_selected_vr_displays', '_inclusive_display_priors': 'finalize_selected_vr_displays__inclusive_display_priors', '_validate_data_production': 'finalize_selected_vr_displays__validate_data_production', '_read_previous_bins': 'finalize_selected_vr_displays__read_previous_bins', '_selected_scans': 'finalize_selected_vr_displays__selected_scans', '_assert_dp_bin_equality': 'finalize_selected_vr_displays__assert_dp_bin_equality', '_apply_dp_audit': 'finalize_selected_vr_displays__apply_dp_audit', '_selected_closure_gate': 'finalize_selected_vr_displays__selected_closure_gate', 'prepare_final_gap_vr': 'finalize_selected_vr_displays_prepare_final_gap_vr'})
_FINALIZE_NAMESPACES['abcd_sideband_data'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_sideband_data_Path', 'gc': 'finalize_sideband_data_gc', 'np': 'finalize_sideband_data_np', 'pd': 'finalize_sideband_data_pd', 'common': 'finalize_sideband_data_common', 'PROTECTED': 'finalize_sideband_data_PROTECTED', 'candidate_coordinates': 'finalize_sideband_data_candidate_coordinates', 'cached_atomic_measurements': 'finalize_sideband_data_cached_atomic_measurements', 'candidate_masks': 'finalize_sideband_data_candidate_masks', 'disclosure_audit': 'finalize_sideband_data_disclosure_audit', 'SIDEBANDS': 'finalize_sideband_data_SIDEBANDS', 'IDENTITY': 'finalize_sideband_data_IDENTITY', 'PREFERRED_ORDER': 'finalize_sideband_data_PREFERRED_ORDER', 'SUMMARY_NUMERIC': 'finalize_sideband_data_SUMMARY_NUMERIC', 'run_nominal_sideband_data': 'finalize_sideband_data_run_nominal_sideband_data', 'data_prediction': 'finalize_sideband_data_data_prediction', 'sideband_indices': 'finalize_sideband_data_sideband_indices', '_load_approved_sidebands': 'finalize_sideband_data__load_approved_sidebands', 'r': 'finalize_sideband_data_r'})
_FINALIZE_NAMESPACES['abcd_sideband_plots'] = _FinalizeNamespace('abcd_finalize_plots', {'textwrap': 'finalize_sideband_plots_textwrap', 'plt': 'finalize_sideband_plots_plt', 'ListedColormap': 'finalize_sideband_plots_ListedColormap', 'LogNorm': 'finalize_sideband_plots_LogNorm', 'Line2D': 'finalize_sideband_plots_Line2D', 'Patch': 'finalize_sideband_plots_Patch', 'Rectangle': 'finalize_sideband_plots_Rectangle', 'np': 'finalize_sideband_plots_np', 'pd': 'finalize_sideband_plots_pd', '_canvas': 'finalize_sideband_plots__canvas', '_caption': 'finalize_sideband_plots__caption', '_style_table': 'finalize_sideband_plots__style_table', 'SCOPES': 'finalize_sideband_plots_SCOPES', 'SCOPE_LABELS': 'finalize_sideband_plots_SCOPE_LABELS', 'STRATEGY_COLORS': 'finalize_sideband_plots_STRATEGY_COLORS', '_table': 'finalize_sideband_plots__table', '_scan': 'finalize_sideband_plots__scan', '_selected': 'finalize_sideband_plots__selected', '_label': 'finalize_sideband_plots__label', '_colors': 'finalize_sideband_plots__colors', '_data_mask': 'finalize_sideband_plots__data_mask', '_approaches': 'finalize_sideband_plots__approaches', '_heatmap': 'finalize_sideband_plots__heatmap', '_scan_figure': 'finalize_sideband_plots__scan_figure', '_selected_support': 'finalize_sideband_plots__selected_support', '_closure_points': 'finalize_sideband_plots__closure_points', '_vr_closure': 'finalize_sideband_plots__vr_closure', '_release_code': 'finalize_sideband_plots__release_code', '_release_audit': 'finalize_sideband_plots__release_audit', '_comparison': 'finalize_sideband_plots__comparison', 'sideband_figure_definitions': 'finalize_sideband_plots_sideband_figure_definitions', 'make_sideband_figures': 'finalize_sideband_plots_make_sideband_figures', '_nominal_data_mask': 'finalize_sideband_plots__nominal_data_mask', '_nominal_sideband_data': 'finalize_sideband_plots__nominal_sideband_data', '_meeting_conclusion': 'finalize_sideband_plots__meeting_conclusion'})
_FINALIZE_NAMESPACES['abcd_sideband_scan_validation'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_sideband_scan_validation_Path', 'gc': 'finalize_sideband_scan_validation_gc', 'hashlib': 'finalize_sideband_scan_validation_hashlib', 'json': 'finalize_sideband_scan_validation_json', 'np': 'finalize_sideband_scan_validation_np', 'pd': 'finalize_sideband_scan_validation_pd', 'threadpool_limits': 'finalize_sideband_scan_validation_threadpool_limits', 'common': 'finalize_sideband_scan_validation_common', 'previous': 'finalize_sideband_scan_validation_previous', 'candidate_coordinates': 'finalize_sideband_scan_validation_candidate_coordinates', 'statistics_from_regions': 'finalize_sideband_scan_validation_statistics_from_regions', 'project_mass_partitions': 'finalize_sideband_scan_validation_project_mass_partitions', '_state_index': 'finalize_sideband_scan_validation__state_index', 'strata_specs': 'finalize_sideband_scan_validation_strata_specs', 'project_strata': 'finalize_sideband_scan_validation_project_strata', 'REGIONS': 'finalize_sideband_scan_validation_REGIONS', 'IDENTITY': 'finalize_sideband_scan_validation_IDENTITY', 'SCOPES': 'finalize_sideband_scan_validation_SCOPES', 'run_scan_vr_validation': 'finalize_sideband_scan_validation_run_scan_vr_validation', 'scan_paths': 'finalize_sideband_scan_validation_scan_paths', '_audit_prior_snapshots': 'finalize_sideband_scan_validation__audit_prior_snapshots', 'load_scan_mc': 'finalize_sideband_scan_validation_load_scan_mc', '_validate_mc_inputs': 'finalize_sideband_scan_validation__validate_mc_inputs', '_joint_atomic': 'finalize_sideband_scan_validation__joint_atomic', '_scope_masks': 'finalize_sideband_scan_validation__scope_masks', '_unique_masks': 'finalize_sideband_scan_validation__unique_masks', '_prior_masks': 'finalize_sideband_scan_validation__prior_masks', '_cached_sources': 'finalize_sideband_scan_validation__cached_sources', '_regress_candidate_mc': 'finalize_sideband_scan_validation__regress_candidate_mc', '_load_inclusive_data': 'finalize_sideband_scan_validation__load_inclusive_data', '_regress_previous_data': 'finalize_sideband_scan_validation__regress_previous_data', '_validate_prior_fine_mc': 'finalize_sideband_scan_validation__validate_prior_fine_mc', '_validate_joint_complement_safety': 'finalize_sideband_scan_validation__validate_joint_complement_safety', 'joint_disclosure_audit': 'finalize_sideband_scan_validation_joint_disclosure_audit', '_exhaustive_guard': 'finalize_sideband_scan_validation__exhaustive_guard', '_guard_projection_squared': 'finalize_sideband_scan_validation__guard_projection_squared', '_guard_probe': 'finalize_sideband_scan_validation__guard_probe', '_guard_commit': 'finalize_sideband_scan_validation__guard_commit', 'data_statistics': 'finalize_sideband_scan_validation_data_statistics', '_validate_final_exhaustive_basis': 'finalize_sideband_scan_validation__validate_final_exhaustive_basis'})
_FINALIZE_NAMESPACES['abcd_sideband_tradeoff'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_sideband_tradeoff_Path', 'json': 'finalize_sideband_tradeoff_json', 'np': 'finalize_sideband_tradeoff_np', 'pd': 'finalize_sideband_tradeoff_pd', 'common': 'finalize_sideband_tradeoff_common', 'review': 'finalize_sideband_tradeoff_review', 'REGIONS': 'finalize_sideband_tradeoff_REGIONS', 'BASELINES': 'finalize_sideband_tradeoff_BASELINES', 'PROTECTED': 'finalize_sideband_tradeoff_PROTECTED', 'run_sideband_tradeoff': 'finalize_sideband_tradeoff_run_sideband_tradeoff', '_read': 'finalize_sideband_tradeoff__read', '_coordinate_name': 'finalize_sideband_tradeoff__coordinate_name', 'candidate_specifications': 'finalize_sideband_tradeoff_candidate_specifications', 'candidate_coordinates': 'finalize_sideband_tradeoff_candidate_coordinates', 'statistics_from_regions': 'finalize_sideband_tradeoff_statistics_from_regions', 'region_rectangles': 'finalize_sideband_tradeoff_region_rectangles', '_overlap': 'finalize_sideband_tradeoff__overlap', 'region_geometry': 'finalize_sideband_tradeoff_region_geometry', '_quantiles': 'finalize_sideband_tradeoff__quantiles', '_pareto': 'finalize_sideband_tradeoff__pareto', 'select_sideband_candidates': 'finalize_sideband_tradeoff_select_sideband_candidates'})
_FINALIZE_NAMESPACES['abcd_sideband_validation'] = _FinalizeNamespace('abcd_finalize_common', {'Path': 'finalize_sideband_validation_Path', 'json': 'finalize_sideband_validation_json', 'np': 'finalize_sideband_validation_np', 'pd': 'finalize_sideband_validation_pd', 'threadpool_limits': 'finalize_sideband_validation_threadpool_limits', 'common': 'finalize_sideband_validation_common', 'candidate_coordinates': 'finalize_sideband_validation_candidate_coordinates', 'statistics_from_regions': 'finalize_sideband_validation_statistics_from_regions', 'REGIONS': 'finalize_sideband_validation_REGIONS', 'MEASUREMENT_FIELDS': 'finalize_sideband_validation_MEASUREMENT_FIELDS', 'SCOPE_DEFINITIONS': 'finalize_sideband_validation_SCOPE_DEFINITIONS', 'run_sideband_validation': 'finalize_sideband_validation_run_sideband_validation', '_key': 'finalize_sideband_validation__key', '_background_index': 'finalize_sideband_validation__background_index', '_signal_index': 'finalize_sideband_validation__signal_index', '_load_caches': 'finalize_sideband_validation__load_caches', '_ratios': 'finalize_sideband_validation__ratios', '_release_order': 'finalize_sideband_validation__release_order', 'cached_atomic_measurements': 'finalize_sideband_validation_cached_atomic_measurements', 'rectangular_mask': 'finalize_sideband_validation_rectangular_mask', 'candidate_masks': 'finalize_sideband_validation_candidate_masks', '_basis': 'finalize_sideband_validation__basis', '_identifiable': 'finalize_sideband_validation__identifiable', '_prospective_patterns': 'finalize_sideband_validation__prospective_patterns', 'disclosure_audit': 'finalize_sideband_validation_disclosure_audit', '_existing_observations': 'finalize_sideband_validation__existing_observations', '_validated_strata_mc': 'finalize_sideband_validation__validated_strata_mc', '_native_atomic_values': 'finalize_sideband_validation__native_atomic_values', '_fine_atomic_measurements': 'finalize_sideband_validation__fine_atomic_measurements', 'r': 'finalize_sideband_validation_r'})
_FINALIZE_NAMESPACES['abcd_wp_comparison'] = _FinalizeNamespace('abcd_finalize_common', {'deepcopy': 'finalize_wp_comparison_deepcopy', 'datetime': 'finalize_wp_comparison_datetime', 'timezone': 'finalize_wp_comparison_timezone', 'json': 'finalize_wp_comparison_json', 'Path': 'finalize_wp_comparison_Path', 'plt': 'finalize_wp_comparison_plt', 'PdfPages': 'finalize_wp_comparison_PdfPages', 'np': 'finalize_wp_comparison_np', 'pd': 'finalize_wp_comparison_pd', 'common': 'finalize_wp_comparison_common', 'plots': 'finalize_wp_comparison_plots', 'review': 'finalize_wp_comparison_review', 'REFERENCE_WP': 'finalize_wp_comparison_REFERENCE_WP', 'WORKING_POINTS': 'finalize_wp_comparison_WORKING_POINTS', '_point_rows': 'finalize_wp_comparison__point_rows', '_sr_view': 'finalize_wp_comparison__sr_view', '_validate_view': 'finalize_wp_comparison__validate_view', '_working_point_view': 'finalize_wp_comparison__working_point_view', 'run_wp_comparison': 'finalize_wp_comparison_run_wp_comparison', '_summary_figures': 'finalize_wp_comparison__summary_figures', 'make_wp_figures': 'finalize_wp_comparison_make_wp_figures', 'export_wp_comparison': 'finalize_wp_comparison_export_wp_comparison'})
_FINALIZE_NAMESPACES['review_mc_sr_working_points'] = _FinalizeNamespace('abcd_finalize_common', {'argparse': 'finalize_review_mc_sr_working_points_argparse', 'hashlib': 'finalize_review_mc_sr_working_points_hashlib', 'json': 'finalize_review_mc_sr_working_points_json', 'Path': 'finalize_review_mc_sr_working_points_Path', 'matplotlib': 'finalize_review_mc_sr_working_points_matplotlib', 'plt': 'finalize_review_mc_sr_working_points_plt', 'PdfPages': 'finalize_review_mc_sr_working_points_PdfPages', 'TwoSlopeNorm': 'finalize_review_mc_sr_working_points_TwoSlopeNorm', 'Rectangle': 'finalize_review_mc_sr_working_points_Rectangle', 'np': 'finalize_review_mc_sr_working_points_np', 'pd': 'finalize_review_mc_sr_working_points_pd', 'REFERENCE_WP': 'finalize_review_mc_sr_working_points_REFERENCE_WP', 'LOCAL_ISO': 'finalize_review_mc_sr_working_points_LOCAL_ISO', 'LOCAL_DPHI': 'finalize_review_mc_sr_working_points_LOCAL_DPHI', 'INPUTS': 'finalize_review_mc_sr_working_points_INPUTS', 'LABELS': 'finalize_review_mc_sr_working_points_LABELS', 'SOURCE': 'finalize_review_mc_sr_working_points_SOURCE', 'asimov_significance': 'finalize_review_mc_sr_working_points_asimov_significance', 'validate_significance': 'finalize_review_mc_sr_working_points_validate_significance', 'file_hash': 'finalize_review_mc_sr_working_points_file_hash', 'finite_ratio': 'finalize_review_mc_sr_working_points_finite_ratio', 'signal_significance': 'finalize_review_mc_sr_working_points_signal_significance', 'summarize_significance': 'finalize_review_mc_sr_working_points_summarize_significance', 'rank_and_select': 'finalize_review_mc_sr_working_points_rank_and_select', 'shortlisted_vr': 'finalize_review_mc_sr_working_points_shortlisted_vr', 'plot_gain_maps': 'finalize_review_mc_sr_working_points_plot_gain_maps', 'plot_signal_gains': 'finalize_review_mc_sr_working_points_plot_signal_gains', 'plot_vr_closure': 'finalize_review_mc_sr_working_points_plot_vr_closure', 'run_review': 'finalize_review_mc_sr_working_points_run_review', 'main': 'finalize_review_mc_sr_working_points_main'})
abcd_4mu_final_09style = _FINALIZE_NAMESPACES['abcd_4mu_final_09style']
abcd_4mu_final_09style_scans = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_scans']
abcd_4mu_final_09style_validation = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_validation']
abcd_4mu_final_data = _FINALIZE_NAMESPACES['abcd_4mu_final_data']
abcd_4mu_final_mc = _FINALIZE_NAMESPACES['abcd_4mu_final_mc']
abcd_4mu_final_report = _FINALIZE_NAMESPACES['abcd_4mu_final_report']
abcd_buffer_distributions = _FINALIZE_NAMESPACES['abcd_buffer_distributions']
abcd_capped_gap_plots = _FINALIZE_NAMESPACES['abcd_capped_gap_plots']
abcd_comprehensive = _FINALIZE_NAMESPACES['abcd_comprehensive']
abcd_comprehensive_plots = _FINALIZE_NAMESPACES['abcd_comprehensive_plots']
abcd_data_expected_significance = _FINALIZE_NAMESPACES['abcd_data_expected_significance']
abcd_final_check = _FINALIZE_NAMESPACES['abcd_final_check']
abcd_final_check_plots = _FINALIZE_NAMESPACES['abcd_final_check_plots']
abcd_final_check_strata = _FINALIZE_NAMESPACES['abcd_final_check_strata']
abcd_final_comparison_plots = _FINALIZE_NAMESPACES['abcd_final_comparison_plots']
abcd_final_nominal_controls = _FINALIZE_NAMESPACES['abcd_final_nominal_controls']
abcd_finalize_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
abcd_finalize_plots = _FINALIZE_NAMESPACES['abcd_finalize_plots']
abcd_inclusive_dp_bins = _FINALIZE_NAMESPACES['abcd_inclusive_dp_bins']
abcd_iso_buffer_plots = _FINALIZE_NAMESPACES['abcd_iso_buffer_plots']
abcd_iso_focus = _FINALIZE_NAMESPACES['abcd_iso_focus']
abcd_local_scan = _FINALIZE_NAMESPACES['abcd_local_scan']
abcd_local_scan_plots = _FINALIZE_NAMESPACES['abcd_local_scan_plots']
abcd_local_scan_statistics = _FINALIZE_NAMESPACES['abcd_local_scan_statistics']
abcd_mass_inclusive = _FINALIZE_NAMESPACES['abcd_mass_inclusive']
abcd_selected_vr_displays = _FINALIZE_NAMESPACES['abcd_selected_vr_displays']
abcd_sideband_data = _FINALIZE_NAMESPACES['abcd_sideband_data']
abcd_sideband_plots = _FINALIZE_NAMESPACES['abcd_sideband_plots']
abcd_sideband_scan_validation = _FINALIZE_NAMESPACES['abcd_sideband_scan_validation']
abcd_sideband_tradeoff = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff']
abcd_sideband_validation = _FINALIZE_NAMESPACES['abcd_sideband_validation']
abcd_wp_comparison = _FINALIZE_NAMESPACES['abcd_wp_comparison']
review_mc_sr_working_points = _FINALIZE_NAMESPACES['review_mc_sr_working_points']


# ========================================================================
# abcd_4mu_final_data: original study API
# ========================================================================
"""Gated reuse of public Data in the 4μ final study; no original Data loader.

Reuse the complete ABCD observations already published by 02/04 only within
the same mass/displacement scope. Each new buffer region must match a public
rectangle exactly or be reconstructed exactly from the public observation
row space. No new linear information is added, so combining these values
with existing observations introduces no new complementary disclosure.
Original nominal DD A/B/C/D and intermediate DD Data are not read.
"""
from pathlib import Path as finalize_4mu_final_data_Path
import hashlib as finalize_4mu_final_data_hashlib
import json as finalize_4mu_final_data_json

import numpy as finalize_4mu_final_data_np
import pandas as finalize_4mu_final_data_pd
from scipy.linalg import qr as finalize_4mu_final_data_qr; from scipy.linalg import solve_triangular as finalize_4mu_final_data_solve_triangular
from threadpoolctl import threadpool_limits as finalize_4mu_final_data_threadpool_limits


finalize_4mu_final_data_REGIONS = tuple('ABCD')
finalize_4mu_final_data_PUBLIC_SCOPES = tuple('low_'+state for state in ('PP','PD','DP','DD')) + tuple(
    'intermediate_'+state for state in ('PP','PD','DP'))
finalize_4mu_final_data_FINAL_A_SPECS = ((.25,2.4),(.30,2.4))
finalize_4mu_final_data_DATA_FIELDS = (list(finalize_4mu_final_data_REGIONS)+['var_'+region for region in finalize_4mu_final_data_REGIONS]
               +[region+'_error' for region in finalize_4mu_final_data_REGIONS]
               +['pred','pred_error','kappa','kappa_error','min_data_count','min_sideband_count'])


def finalize_4mu_final_data_build_data_study(notebook_dir,mc_result):
    """Return public-only VR closure tables and explicit nominal Data status."""
    with finalize_4mu_final_data_threadpool_limits(limits=1,user_api='blas'):
        return finalize_4mu_final_data__build(finalize_4mu_final_data_Path(notebook_dir).resolve(),mc_result)


def finalize_4mu_final_data__build(notebook_dir,mc_result):
    public,sources,checks = finalize_4mu_final_data__public_inputs(notebook_dir)
    candidates=mc_result['tables']['candidates'].copy()
    if candidates.candidate_id.duplicated().any():
        raise AssertionError('Each candidate_id must be unique.')
    regions_audit,closure,basis_rows,nominal_rows = [],[],[],[]
    basis_cache={}
    for scope,records in public.items():
        reference=mc_result['backgrounds'][scope]
        atoms=finalize_4mu_final_data__atoms(reference)
        masks,counts,tokens=finalize_4mu_final_data__public_masks(records,atoms)
        basis,responses,rank=finalize_4mu_final_data__basis(masks,counts,width=len(atoms))
        basis_cache[scope]=(atoms,basis,responses,tokens)
        basis_rows.append(dict(scope=scope,public_threshold_points=len(records),
                               public_region_masks=len(masks),rank=rank,atomic_cells=len(atoms)))
    for scope,background in mc_result['backgrounds'].items():
        signals=mc_result['signals'][scope]
        if len(signals)!=120:
            raise AssertionError('Each scope requires 120 separate signal hypotheses.')
        definition=mc_result['scopes'][scope]
        state=definition.get('state',definition.get('plane'))
        if state is None:
            state=scope.rsplit('_',1)[-1]
        for spec in candidates.to_dict('records'):
            identity={key:spec.get(key) for key in
                      ('candidate_id','strategy','anchor','a_iso','a_phi','b_iso','c_phi')}
            identity.update(scope=scope,plane=state,mass_slice=definition.get('mass_slice'),
                            mass_low=definition['mass_low'],mass_high=definition['mass_high'])
            masks=finalize_4mu_final_data__candidate_masks(background,spec)
            gate=finalize_4mu_final_data__gates(background,signals,masks)
            is_nominal=scope=='nominal_DD'
            own_gate=all(row['purity_pass'] for row in gate
                         if not is_nominal or row['region'] in 'BCD')
            values={}
            geometry_safe=finalize_4mu_final_data__nominal_geometry(background,masks) if is_nominal else True
            for region,item in zip(finalize_4mu_final_data_REGIONS,gate):
                exact,known,residual=False,False,finalize_4mu_final_data_np.nan
                if scope in basis_cache and own_gate:
                    atoms,basis,responses,tokens=basis_cache[scope]
                    query=finalize_4mu_final_data__rectangle(atoms,*finalize_4mu_final_data__bounds(spec,region))
                    exact=finalize_4mu_final_data_np.packbits(query).tobytes() in tokens
                    coefficients=basis.T @ query.astype(float)
                    residual=float(finalize_4mu_final_data_np.max(finalize_4mu_final_data_np.abs(query.astype(float)-basis @ coefficients),initial=0.))
                    known=residual<1e-8
                    if known:
                        value=float(coefficients @ responses)
                        if value < -1e-6 or not finalize_4mu_final_data_np.isclose(value,round(value),atol=1e-6):
                            raise AssertionError('Reconstructed public Data must be nonnegative integers.')
                        values[region]=float(round(value))
                regions_audit.append(dict(identity,**item,exact_previous_rectangle=exact,
                    public_row_space_known=known,row_space_residual=residual,
                    original_data_queried=False,new_public_linear_functionals=0,
                    protection='Exact reuse of existing public observations; no combination across scopes'))
            available=bool(scope in finalize_4mu_final_data_PUBLIC_SCOPES and own_gate and len(values)==4)
            reason=('EXACT_PUBLIC_REGIONS_OR_ROW_SPACE_REUSE' if available else
                    'NOMINAL_MC_GATE_FAILED_NO_DATA_QUERY' if is_nominal and not own_gate else
                    'NOMINAL_FINAL_A_OVERLAP_NO_DATA_QUERY' if is_nominal and not geometry_safe else
                    'NO_PUBLISHED_NOMINAL_CONTROL_COUNTS' if is_nominal else
                    'DD_NOMINAL_MASS_DATA_PROTECTED' if state=='DD' and scope!='low_DD' else
                    'NO_PUBLIC_SAME_SELECTION' if scope not in finalize_4mu_final_data_PUBLIC_SCOPES else
                    'VR_MC_GATE_FAILED' if not own_gate else
                    'BUFFER_REGION_NOT_IN_PUBLIC_ROW_SPACE')
            row=dict(identity,source='DATA',approved=own_gate,purity_pass=own_gate,
                     display_allowed=available,reason=reason,
                     max_S_B=max((item['max_S_B'] for item in gate
                                  if (not is_nominal or item['region'] in 'BCD')
                                  and finalize_4mu_final_data_np.isfinite(item['max_S_B'])),default=finalize_4mu_final_data_np.nan),
                     nominal_geometry_safe=geometry_safe,original_data_queried=False,
                     new_public_linear_functionals=0,DATA_A_evaluated=False,
                     status='PUBLIC_DATA_REUSE' if available else 'DATA_NOT_EVALUATED')
            row.update(finalize_4mu_final_data__statistics(values) if available else finalize_4mu_final_data__blank_statistics())
            closure.append(row)
            if is_nominal:
                nominal_rows.append(dict(identity,mc_BCD_gate_pass=own_gate,
                    final_A_geometry_safe=geometry_safe,protected_A='union of reviewed A(.25,2.4),A(.30,2.4)',
                    data_B=finalize_4mu_final_data_np.nan,data_C=finalize_4mu_final_data_np.nan,data_D=finalize_4mu_final_data_np.nan,pred=finalize_4mu_final_data_np.nan,pred_error=finalize_4mu_final_data_np.nan,
                    DATA_A_evaluated=False,original_data_queried=False,reason=reason))
    frame=finalize_4mu_final_data_pd.DataFrame(closure)
    hidden=frame.loc[~frame.display_allowed]
    if not hidden[finalize_4mu_final_data_DATA_FIELDS].isna().all().all():
        raise AssertionError('All blocked Data values must be NaN.')
    nominal=finalize_4mu_final_data_pd.DataFrame(nominal_rows)
    if nominal.mc_BCD_gate_pass.any():
        # A future passing nominal candidate remains unqueried until the parent
        # explicitly agrees a prospective, jointly protected query design.
        checks.append(dict(scope='nominal_DD',check='Gate-passing nominal remains public-only/unqueried',passed=True))
    finalize_4mu_final_data__regress_selected(frame,notebook_dir,checks)
    bins,bin_conservation=finalize_4mu_final_data__public_data_bins(mc_result,basis_cache)
    checks.extend([
        dict(scope='DATA',check='No original Data loader; all values reuse the row space of existing same-scope public observations',passed=True),
        dict(scope='Disclosure',check='Zero new linear information; combining with any existing release adds no unsafe complement',passed=True),
        dict(scope='Nominal',check='No original or new nominal DD A/B/C/D counts are queried',passed=True),
        dict(scope='Ordering',check='4μ leading/subleading LJ order and partner isolation [0,.5) are preserved',passed=True),
        dict(scope='Data 2D',check='Bin gate and same-scope public row space are checked independently; masked bins are NaN and observed zeros stay zero',passed=True),
        dict(scope='2D flow',check='154 folded bins preserve native physical-range sumw/sumw2 for the background and each of 120 signals',passed=bool(bin_conservation.passed.all()))])
    return dict(tables=dict(data_closure=frame,data_region_audit=finalize_4mu_final_data_pd.DataFrame(regions_audit),
                           data_nominal_status=nominal,data_public_basis=finalize_4mu_final_data_pd.DataFrame(basis_rows),
                           data_prior_sources=finalize_4mu_final_data_pd.DataFrame(sources),data_validation=finalize_4mu_final_data_pd.DataFrame(checks),
                           data_bins=bins,data_bin_conservation=bin_conservation),
                metadata=dict(DATA_A_evaluated=False,nominal_DATA_A_evaluated=False,
                    nominal_DATA_BCD_evaluated=False,original_data_files_opened=0,
                    original_data_files_hashed=0,new_public_linear_functionals=0,
                    new_data_queries=0,purity_signal_hypotheses=120,
                    public_threshold_points=sum(len(rows) for rows in public.values()),
                    displayed_vr_closures=int(frame.display_allowed.sum()),
                    public_only=True,scope_mixing=False,
                    nominal_control_prediction_available=False,
                    displayed_data_2d_bins=int(bins.display_allowed.sum()),
                    data_2d_total_bins=len(bins),data_2d_integrated_gate_independent=True,
                    data_2d_tail='Displayed [1,2) represents physical [1,∞); angle underflow/overflow is folded into the first/last displayed bin',
                    final_A_protection='reviewed WP1/WP2 union I0<.30, |dphi|>=2.4; not historical candidate union',
                    missing_data_policy='Retain blocking reasons and NaN without new queries'))


def finalize_4mu_final_data__public_inputs(notebook_dir):
    public,sources,checks={},[],[]
    original_manifests=[]
    for folder,window,low,high,states in (
            ('02_4mu_VR1_summary_outputs','low',0.,150.,('PP','PD','DP','DD')),
            ('04_4mu_VR2_summary_outputs','intermediate',150.,300.,('PP','PD','DP'))):
        directory=notebook_dir/folder
        manifest=finalize_4mu_final_data_json.loads((directory/'manifest.json').read_text())
        settings=manifest['settings']
        if (settings['topology'],settings['current_hist'],settings['mass_min'],settings['mass_max'],
                settings['partner_iso_low'],settings['partner_iso_high'],settings['purity_limit']) != (
                    '4mu','abcd6d_4mu',low,high,0.,.5,.1):
            raise AssertionError('The public 4μ selection differs from the current definition.')
        grid=finalize_4mu_final_data_pd.read_csv(directory/'grid.csv',float_precision='round_trip')
        data=grid.loc[grid.source.eq('DATA')]
        if not set(data.plane.unique()).issubset(states):
            raise AssertionError('The previously published table contains an unexpected displacement state.')
        selected=data.loc[data.approved.eq(True)]
        if not selected.signal_hypotheses.eq(120).all():
            raise AssertionError('Previously published observations also require the 120-model gate.')
        values=selected[list('ABCD')].to_numpy(float)
        variances=selected[['var_'+region for region in finalize_4mu_final_data_REGIONS]].to_numpy(float)
        if (not finalize_4mu_final_data_np.isfinite(values).all() or (values<0).any()
                or not finalize_4mu_final_data_np.array_equal(values,finalize_4mu_final_data_np.rint(values)) or not finalize_4mu_final_data_np.array_equal(values,variances)):
            raise AssertionError('Previously published Data observations require integer counts and variance=N.')
        for state in states:
            public[window+'_'+state]=selected.loc[selected.plane.eq(state)].to_dict('records')
        for name in ('manifest.json','grid.csv','scan.csv','input_manifest.csv','figure_manifest.csv'):
            path=directory/name
            sources.append(finalize_4mu_final_data__record(path,'Previously published 4μ same-selection Data/provenance'))
        original=finalize_4mu_final_data_pd.read_csv(directory/'input_manifest.csv')
        original=original.loc[original.source.eq('DATA selected VR candidates')]
        if len(original)!=68:
            raise AssertionError('The established 4μ Data production requires 68 original files.')
        original_manifests.append(original[['path','size_bytes','sha256']].sort_values('path').reset_index(drop=True))
        checks.append(dict(scope=folder,check='Public Data counts and all120 gate verified for identical mass, displacement and partner isolation',passed=True))
    finalize_4mu_final_data_pd.testing.assert_frame_equal(*original_manifests,check_dtype=False)
    checks.append(dict(scope='Provenance',check='02/04 use identical path/size/SHA256 metadata for 68 original Data files; originals remain unopened',passed=True))
    # Notebook 08 pins the original 04 CSV and manifest hashes, giving an
    # independent already-existing checksum reference for the public grid.
    pinned=finalize_4mu_final_data_pd.read_csv(notebook_dir/'08_4mu_VR2_focused_outputs'/'input_manifest.csv')
    for row in pinned.itertuples():
        path=finalize_4mu_final_data_Path(row.path)
        if path.suffix not in ('.csv','.json'):
            raise AssertionError('The 08 provenance check may not open original Data.')
        record=finalize_4mu_final_data__record(path,'Public 04 source hash pinned by 08')
        if (record['sha256'],record['size_bytes'])!=(row.sha256,row.size_bytes):
            raise AssertionError('The public 04 cache differs from the input manifest pinned by 08.')
    primary=finalize_4mu_final_data_pd.read_csv(notebook_dir/'04_4mu_VR2_summary_outputs'/'grid.csv',float_precision='round_trip')
    for folder,name in (('06_4mu_WP_comparison_outputs','shared_grid.csv'),
                        ('08_4mu_VR2_focused_outputs','full_vr2_closure.csv')):
        path=notebook_dir/folder/name
        if folder=='06_4mu_WP_comparison_outputs' and not path.is_file():
            manifest_path=notebook_dir/folder/'manifest.json'
            maintenance=finalize_4mu_final_data_json.loads(manifest_path.read_text()).get('export_maintenance',{})
            aliases=[row for row in maintenance.get('canonical_exports',[]) if row.get('alias')==name]
            canonical=notebook_dir/'04_4mu_VR2_summary_outputs'/'grid.csv'
            if (len(aliases)!=1 or not canonical.is_file()
                    or finalize_4mu_final_data_Path(aliases[0].get('canonical_path','')).resolve()!=canonical.resolve()):
                raise AssertionError('The 06 shared grid requires a preservation record for an identical canonical CSV.')
            record=finalize_4mu_final_data__record(canonical,'Canonical identity check for the 06 shared grid')
            if (record['sha256'],record['size_bytes'])!=(aliases[0].get('sha256'),aliases[0].get('size_bytes')):
                raise AssertionError('The preserved hash/size differs between the 06 shared grid and canonical CSV.')
            sources.append(finalize_4mu_final_data__record(manifest_path,'Canonical export preservation record for the 06 shared grid'))
            path=canonical
        other=finalize_4mu_final_data_pd.read_csv(path,float_precision='round_trip')
        fields=['plane','source','tx','ty','approved']+list(finalize_4mu_final_data_REGIONS)+['var_'+region for region in finalize_4mu_final_data_REGIONS]
        left=primary.loc[primary.source.eq('DATA'),fields].sort_values(['plane','tx','ty']).reset_index(drop=True)
        right=other.loc[other.source.eq('DATA'),fields].sort_values(['plane','tx','ty']).reset_index(drop=True)
        finalize_4mu_final_data_pd.testing.assert_frame_equal(left,right,check_dtype=False,rtol=1e-10,atol=1e-10)
        sources.append(finalize_4mu_final_data__record(path,'Regression check of 06/08 reuse of public 04 observations'))
        checks.append(dict(scope=folder,check='Regression check confirms reuse of the same public Data observations as 04',passed=True))
    return public,sources,checks


def finalize_4mu_final_data__record(path,role):
    return dict(source=role,path=str(path),size_bytes=path.stat().st_size,
                sha256=finalize_4mu_final_data_hashlib.sha256(path.read_bytes()).hexdigest(),original_data_file=False)


def finalize_4mu_final_data__atoms(plane):
    x=finalize_4mu_final_data_np.r_[plane['xedges'],finalize_4mu_final_data_np.inf]
    y=finalize_4mu_final_data_np.r_[-finalize_4mu_final_data_np.inf,plane['yedges'],finalize_4mu_final_data_np.inf]
    return finalize_4mu_final_data_pd.DataFrame([dict(xlow=x[i],xhigh=x[i+1],ylow=y[j],yhigh=y[j+1])
                         for i in range(len(x)-1) for j in range(len(y)-1)])


def finalize_4mu_final_data__rectangle(atoms,xlow,xhigh,ylow,yhigh):
    return ((atoms.xlow>=xlow)&(atoms.xhigh<=xhigh)
            &(atoms.ylow>=ylow)&(atoms.yhigh<=yhigh)).to_numpy(bool)


def finalize_4mu_final_data__bounds(spec,region):
    low=region in 'AC'
    high=region in 'AB'
    return (0. if low else spec['b_iso'],spec['a_iso'] if low else finalize_4mu_final_data_np.inf,
            spec['a_phi'] if high else -finalize_4mu_final_data_np.inf,finalize_4mu_final_data_np.inf if high else spec['c_phi'])


def finalize_4mu_final_data__public_masks(records,atoms):
    masks,counts,seen=[],[],{}
    for row in records:
        spec=dict(a_iso=row['tx'],b_iso=row['tx'],a_phi=row['ty'],c_phi=row['ty'])
        for region in finalize_4mu_final_data_REGIONS:
            mask=finalize_4mu_final_data__rectangle(atoms,*finalize_4mu_final_data__bounds(spec,region))
            token=finalize_4mu_final_data_np.packbits(mask).tobytes()
            value=float(row[region])
            if token in seen:
                finalize_4mu_final_data_np.testing.assert_allclose(seen[token],value,rtol=0,atol=1e-8)
            elif mask.any():
                seen[token]=value;masks.append(mask);counts.append(value)
    return masks,finalize_4mu_final_data_np.asarray(counts),set(seen)


def finalize_4mu_final_data__basis(masks,counts,width=None):
    matrix=finalize_4mu_final_data_np.asarray(masks,float)
    if not len(matrix):
        return finalize_4mu_final_data_np.empty((0 if width is None else width,0)),finalize_4mu_final_data_np.empty(0),0
    q,r,pivots=finalize_4mu_final_data_qr(matrix.T,mode='economic',pivoting=True)
    rank=int(finalize_4mu_final_data_np.count_nonzero(finalize_4mu_final_data_np.abs(finalize_4mu_final_data_np.diag(r))>1e-9))
    basis=q[:,:rank]
    responses=finalize_4mu_final_data_solve_triangular(r[:rank,:rank].T,counts[pivots[:rank]],lower=True)
    finalize_4mu_final_data_np.testing.assert_allclose(matrix @ basis @ responses,counts,rtol=0,atol=1e-6)
    return basis,responses,rank


def finalize_4mu_final_data__candidate_masks(plane,spec):
    x=finalize_4mu_final_data_np.r_[-finalize_4mu_final_data_np.inf,plane['xedges']][:,None]
    y=finalize_4mu_final_data_np.r_[-finalize_4mu_final_data_np.inf,plane['yedges']][None,:]
    return dict(A=(x>=0)&(x<spec['a_iso'])&(y>=spec['a_phi']),
                B=(x>=spec['b_iso'])&(y>=spec['a_phi']),
                C=(x>=0)&(x<spec['a_iso'])&(y<spec['c_phi']),
                D=(x>=spec['b_iso'])&(y<spec['c_phi']))


def finalize_4mu_final_data__gates(background,signals,masks):
    names=sorted(signals)
    result=[]
    for region,mask in masks.items():
        b=float(background['values'][mask].sum())
        v=float(background['variances'][mask].sum())
        s=finalize_4mu_final_data_np.asarray([signals[name]['values'][mask].sum() for name in names])
        valid=bool(b>1e-8 and v>=0 and finalize_4mu_final_data_np.isfinite([b,v]).all() and finalize_4mu_final_data_np.isfinite(s).all() and (s>=0).all())
        worst=int(finalize_4mu_final_data_np.argmax(s));maximum=float(s[worst]/b) if valid else finalize_4mu_final_data_np.nan
        result.append(dict(region=region,background=b,var_background=v,
                           max_S_B=maximum,worst_sample=names[worst],signal_hypotheses=120,
                           valid_background=valid,purity_pass=bool(valid and maximum<.1)))
    return result


def finalize_4mu_final_data__nominal_geometry(plane,masks):
    x=finalize_4mu_final_data_np.r_[-finalize_4mu_final_data_np.inf,plane['xedges']][:,None]
    y=finalize_4mu_final_data_np.r_[-finalize_4mu_final_data_np.inf,plane['yedges']][None,:]
    protected=finalize_4mu_final_data_np.zeros_like(plane['values'],dtype=bool)
    for isolation,angle in finalize_4mu_final_data_FINAL_A_SPECS:
        protected |= (x>=0)&(x<isolation)&(y>=angle)
    return not any((masks[region]&protected).any() for region in 'BCD')


def finalize_4mu_final_data__blank_statistics():
    return {name:finalize_4mu_final_data_np.nan for name in finalize_4mu_final_data_DATA_FIELDS}


def finalize_4mu_final_data__statistics(values):
    result=dict(values)
    for region in finalize_4mu_final_data_REGIONS:
        result['var_'+region]=values[region]
        result[region+'_error']=finalize_4mu_final_data_np.sqrt(values[region])
    a,b,c,d=(values[region] for region in finalize_4mu_final_data_REGIONS)
    result.update(pred=finalize_4mu_final_data_np.nan,pred_error=finalize_4mu_final_data_np.nan,kappa=finalize_4mu_final_data_np.nan,kappa_error=finalize_4mu_final_data_np.nan,
                  min_data_count=min(values.values()),min_sideband_count=min(b,c,d))
    if d<=0:return result
    pred=b*c/d;result['pred']=pred
    if b<=0 or c<=0:return result
    variance=pred*pred*(1/b+1/c+1/d)
    result.update(pred_error=finalize_4mu_final_data_np.sqrt(variance),kappa=a/pred,
                  kappa_error=finalize_4mu_final_data_np.sqrt(a/pred**2+a*a*variance/pred**4))
    return result


def finalize_4mu_final_data__regress_selected(frame,notebook_dir,checks):
    previous=finalize_4mu_final_data_pd.read_csv(notebook_dir/'08_4mu_VR2_focused_outputs'/'selected_closure.csv')
    previous=previous.loc[previous.source.eq('DATA') & previous.wp.isin(['WP1','WP2'])]
    for row in previous.itertuples():
        found=frame.loc[frame.scope.eq('intermediate_'+row.plane)
                        & finalize_4mu_final_data_np.isclose(frame.a_iso,row.tx)&finalize_4mu_final_data_np.isclose(frame.b_iso,row.tx)
                        &finalize_4mu_final_data_np.isclose(frame.a_phi,row.ty)&finalize_4mu_final_data_np.isclose(frame.c_phi,row.ty)]
        if found.empty:continue
        for item in found.itertuples():
            if not item.display_allowed:raise AssertionError('The existing WP1/WP2 Data validation was not reused.')
            fields=list(finalize_4mu_final_data_REGIONS)+['pred','pred_error','kappa','kappa_error']
            finalize_4mu_final_data_np.testing.assert_allclose([getattr(item,name) for name in fields],
                                       [getattr(row,name) for name in fields],rtol=1e-10,atol=1e-10)
    checks.append(dict(scope='08 WP1/WP2',check='Public PP/PD/DP yields, predictions and κ reproduced for identical contiguous candidates',passed=True))


def finalize_4mu_final_data__public_data_bins(mc_result,basis_cache):
    """Redisplay only MC-pure physical bins already in the public row space.

    Integrated candidate closure and a bin's display eligibility are distinct.
    A partial plane never supplies missing ABCD counts or authorizes a closure.
    """
    rows,conservation=[],[]
    for scope in finalize_4mu_final_data_PUBLIC_SCOPES:
        plane=mc_result['backgrounds'][scope]
        signals=mc_result['signals'][scope]
        atoms,basis,responses,_=basis_cache[scope]
        names=sorted(signals)
        if len(names)!=120:
            raise AssertionError('The 2D bin check also requires 120 signal hypotheses.')
        ny=len(plane['yedges'])+1
        # _atoms omits only isolation underflow. Its i-major/j-minor order must
        # be identical to flattening every physical x row and all angle flows.
        finalize_4mu_final_data_np.testing.assert_array_equal(atoms.xlow.to_numpy(),finalize_4mu_final_data_np.repeat(plane['xedges'],ny))
        finalize_4mu_final_data_np.testing.assert_array_equal(atoms.ylow.to_numpy(),
                                       finalize_4mu_final_data_np.tile(finalize_4mu_final_data_np.r_[-finalize_4mu_final_data_np.inf,plane['yedges']],len(plane['xedges'])))
        background=plane['values'][1:,:].ravel()
        background_variance=plane['variances'][1:,:].ravel()
        signal=finalize_4mu_final_data_np.asarray([signals[name]['values'][1:,:].ravel() for name in names])
        signal_variance=finalize_4mu_final_data_np.asarray([signals[name]['variances'][1:,:].ravel() for name in names])
        if (len(background)!=len(atoms) or signal.shape!=(120,len(atoms))
                or signal_variance.shape!=signal.shape):
            raise AssertionError('MC flow arrays and public atomic geometry have different ordering.')
        xedges=finalize_4mu_final_data_np.asarray(plane['xedges'])
        xedges=xedges[xedges<=2.]
        yedges=finalize_4mu_final_data_np.asarray(plane['yedges'])
        if len(xedges)!=12 or len(yedges)!=15:
            raise AssertionError('The established 4μ Data display requires 11×14=154 native-folded bins.')
        cells,masks=[],[]
        for ix,(xlow,xhigh) in enumerate(zip(xedges[:-1],xedges[1:])):
            for iy,(ylow,yhigh) in enumerate(zip(yedges[:-1],yedges[1:])):
                physical_xhigh=finalize_4mu_final_data_np.inf if ix==len(xedges)-2 else xhigh
                physical_ylow=-finalize_4mu_final_data_np.inf if iy==0 else ylow
                physical_yhigh=finalize_4mu_final_data_np.inf if iy==len(yedges)-2 else yhigh
                cells.append(dict(ix=ix,iy=iy,bin_x=ix,bin_y=iy,
                                  xlow=xlow,xhigh=xhigh,ylow=ylow,yhigh=yhigh,
                                  physical_xlow=xlow,physical_xhigh=physical_xhigh,
                                  physical_ylow=physical_ylow,physical_yhigh=physical_yhigh))
                masks.append(finalize_4mu_final_data__rectangle(atoms,xlow,physical_xhigh,physical_ylow,physical_yhigh))
        matrix=finalize_4mu_final_data_np.asarray(masks,float)
        if not finalize_4mu_final_data_np.array_equal(matrix.sum(axis=0),finalize_4mu_final_data_np.ones(len(atoms))):
            raise AssertionError('The 154 folded bins do not partition each native physical cell exactly once.')
        b=matrix @ background;v=matrix @ background_variance
        s=signal @ matrix.T;sv=signal_variance @ matrix.T
        conservation.append(dict(scope=scope,source='Background',sample='Total background',
            sumw_native=float(background.sum()),sumw_folded=float(b.sum()),
            sumw2_native=float(background_variance.sum()),sumw2_folded=float(v.sum()),
            passed=bool(finalize_4mu_final_data_np.allclose([b.sum(),v.sum()],[background.sum(),background_variance.sum()],
                                    rtol=1e-10,atol=1e-8))))
        for index,name in enumerate(names):
            conservation.append(dict(scope=scope,source='Signal',sample=name,
                sumw_native=float(signal[index].sum()),sumw_folded=float(s[index].sum()),
                sumw2_native=float(signal_variance[index].sum()),sumw2_folded=float(sv[index].sum()),
                passed=bool(finalize_4mu_final_data_np.allclose([s[index].sum(),sv[index].sum()],
                                       [signal[index].sum(),signal_variance[index].sum()],rtol=1e-10,atol=1e-8))))
        coefficients=matrix @ basis
        residual=finalize_4mu_final_data_np.max(finalize_4mu_final_data_np.abs(matrix-coefficients @ basis.T),axis=1)
        known=residual<1e-8
        definition=mc_result['scopes'][scope]
        state=definition.get('state',definition.get('plane',scope.rsplit('_',1)[-1]))
        for index,cell in enumerate(cells):
            valid_background=bool(finalize_4mu_final_data_np.isfinite([b[index],v[index]]).all() and b[index]>1e-8 and v[index]>=0)
            valid_signal=bool(finalize_4mu_final_data_np.isfinite(s[:,index]).all() and finalize_4mu_final_data_np.isfinite(sv[:,index]).all()
                              and (s[:,index]>=0).all() and (sv[:,index]>=0).all())
            worst=int(finalize_4mu_final_data_np.argmax(s[:,index]))
            maximum=float(s[worst,index]/b[index]) if valid_background and valid_signal else finalize_4mu_final_data_np.nan
            gate=bool(valid_background and valid_signal and maximum<.1)
            allowed=bool(gate and known[index])
            value=finalize_4mu_final_data_np.nan
            if allowed:
                value=float(coefficients[index] @ responses)
                if value < -1e-6 or not finalize_4mu_final_data_np.isclose(value,round(value),atol=1e-6):
                    raise AssertionError('Reconstructed Data bins must contain nonnegative integer counts.')
                value=float(round(value))
            rows.append(dict(scope=scope,plane=state,source='DATA',
                mass_slice=definition['mass_slice'],mass_low=definition['mass_low'],mass_high=definition['mass_high'],
                **cell,background=float(b[index]),var_background=float(v[index]),
                background_error=float(finalize_4mu_final_data_np.sqrt(v[index])),
                neff_background=float(b[index]**2/v[index]) if v[index]>0 else finalize_4mu_final_data_np.nan,
                signal_max=float(s[worst,index]),var_signal_max=float(sv[worst,index]),
                signal_error=float(finalize_4mu_final_data_np.sqrt(sv[worst,index])),worst_sample=names[worst],
                max_S_B=maximum,signal_hypotheses=120,valid_background=valid_background,
                bin_purity_pass=gate,public_row_space_known=bool(known[index]),
                row_space_residual=float(residual[index]),display_allowed=allowed,
                data=value,var_data=value,data_error=float(finalize_4mu_final_data_np.sqrt(value)) if allowed else finalize_4mu_final_data_np.nan,
                reason='BIN_PURITY_AND_EXACT_PUBLIC_ROW_SPACE_REUSE' if allowed else
                       'BIN_MC_GATE_FAILED' if not gate else 'PUBLIC_BIN_NOT_IDENTIFIABLE',
                integrated_candidate_gate_applied=False,
                original_data_queried=False,new_public_linear_functionals=0))
    bins=finalize_4mu_final_data_pd.DataFrame(rows)
    audit=finalize_4mu_final_data_pd.DataFrame(conservation)
    if not audit.passed.all():
        raise AssertionError('2D folding must preserve sumw/sumw2 for the background and each signal.')
    if not bins.loc[~bins.display_allowed,['data','var_data']].isna().all().all():
        raise AssertionError('Data bins not allowed for display must be NaN.')
    if not bins.loc[bins.display_allowed,'data'].equals(bins.loc[bins.display_allowed,'var_data']):
        raise AssertionError('The variance of each displayed Data bin must equal its count.')
    return bins,audit


# ========================================================================
# abcd_4mu_final_report: original study API
# ========================================================================
"""Notebook 10 orchestration, explicit eligibility decisions and slide export."""
from datetime import datetime as finalize_4mu_final_report_datetime; from datetime import timezone as finalize_4mu_final_report_timezone
import hashlib as finalize_4mu_final_report_hashlib
import json as finalize_4mu_final_report_json
from pathlib import Path as finalize_4mu_final_report_Path

import matplotlib.pyplot as finalize_4mu_final_report_plt
from matplotlib.backends.backend_pdf import PdfPages as finalize_4mu_final_report_PdfPages
import numpy as finalize_4mu_final_report_np
import pandas as finalize_4mu_final_report_pd


finalize_4mu_final_report_SECTION_LABELS = {
    'overview': 'Starting point from 08 and nominal MC statistics',
    'loose': 'Detailed loose-boundary study',
    'buffer': 'Detailed fixed-A buffer study',
    'validation': 'MC and public Data VR validation with identical selections',
    'comparison': 'Comparison of both approaches and finalization assessment',
    'backup': 'Additional diagnostics',
}


def finalize_4mu_final_report_build_4mu_final_report(notebook_dir):
    """Build the complete study without changing a source or opening raw DATA."""
    build_mc_study = _FINALIZE_NAMESPACES['abcd_4mu_final_mc'].build_mc_study
    build_data_study = _FINALIZE_NAMESPACES['abcd_4mu_final_data'].build_data_study
    make_09_style_figures = _FINALIZE_NAMESPACES['abcd_4mu_final_09style'].make_09_style_figures

    notebook_dir = finalize_4mu_final_report_Path(notebook_dir).resolve()
    output = notebook_dir / '10_4mu_final_check_outputs'
    protected = finalize_4mu_final_report_protected_inputs(notebook_dir)
    before = {str(path): finalize_4mu_final_report_file_hash(path) for path in protected}
    finalize_4mu_final_report_plt.rcParams.update({'font.size': 11, 'axes.titlesize': 'large',
        'axes.labelsize': 'medium', 'legend.fontsize': 'medium', 'figure.dpi': 100,
        'axes.spines.top': True, 'axes.spines.right': True,
        'figure.max_open_warning': 0, 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    print('4μ MC: calculating 20 loose points and WP2/WP1 fixed-A buffers.', flush=True)
    result = build_mc_study(notebook_dir)
    result['output_dir'] = output
    result['plot_candidates'] = finalize_4mu_final_report_diagnostic_candidates(result)
    result['tables']['plot_candidates'] = result['plot_candidates'].copy()
    # Diagnostic roles are fixed before loading any public DATA counts.
    print('Reusing only existing public observations from the identical VR.', flush=True)
    data = build_data_study(notebook_dir, result)
    if set(data['tables']).intersection(result['tables']):
        raise AssertionError('MC and DATA table names must be distinct.')
    result['tables'].update(data['tables'])
    result['metadata']['data'] = data['metadata']
    result['tables'].update(finalize_4mu_final_report_summary_tables(notebook_dir, result))
    print('Rendering detailed loose/buffer figures.', flush=True)
    plots = make_09_style_figures(result)
    result['figures'] = plots['figures']
    result['captions'] = plots['captions']
    result['tables'].update(plots.get('tables', {}))
    result['metadata']['presentation'] = plots['metadata']
    after = {str(path): finalize_4mu_final_report_file_hash(path) for path in protected}
    if before != after:
        raise AssertionError('An original notebook, helper or published input changed.')
    result['metadata'].update(source_files_unchanged=True, source_sha256=before,
        nominal_DATA_A_evaluated=False, nominal_DATA_BCD_evaluated=False,
        nominal_DATA_prediction_available=False, final_likelihood_evaluated=False,
        selected_method=None, finalization_status='NO_MC_SUPPORTED_NOMINAL_CONTROL_DEFINITION',
        data_used_for_candidate_ranking=False,
        diagnostic_candidate_policy='Loose: maximum absolute median known-background MC-A Z among the 20 boundary-scan points before the purity gate; diagnostic only. Buffer: fixed WP2 A, minimum supported control contamination then minimum native movement. No Data ranking or adopted method.',
        diagram_conventions='leading μ-LJ first, subleading μ-LJ second; all model hypotheses separate')
    if result['tables']['mc_nominal_scan'].purity_pass_BCD.any():
        result['metadata']['finalization_status'] = 'ELIGIBLE_MC_CANDIDATE_REQUIRES_COMPLETE_REVIEW'
    checks = [result['tables']['validation'].assign(component='MC'),
              result['tables']['data_validation'].assign(component='public_DATA')]
    if 'data_bin_conservation' in result['tables']:
        checks.append(result['tables']['data_bin_conservation'].assign(component='display_binning'))
    result['tables']['validation_all'] = finalize_4mu_final_report_pd.concat(checks, ignore_index=True)
    assert result['tables']['validation_all'].passed.all()
    finalize_4mu_final_report_save_report(result, notebook_dir)
    for _, figure in result['figures']:
        finalize_4mu_final_report_plt.close(figure)
    print(f"Complete: {len(result['figures'])} figures. "
          f"Nominal gate passes: {int(result['tables']['mc_nominal_scan'].purity_pass_BCD.sum())}; "
          'the assessment table records the finalization status.', flush=True)
    return result


def finalize_4mu_final_report_file_hash(path):
    digest = finalize_4mu_final_report_hashlib.sha256()
    with finalize_4mu_final_report_Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_4mu_final_report_protected_inputs(notebook_dir):
    paths = []
    for prefix in ('02_4mu_VR1', '04_4mu_VR2', '06_4mu_WP', '08_4mu_VR2'):
        for folder in notebook_dir.glob(prefix+'*outputs'):
            paths.extend(sorted(folder.glob('*.csv')))
            paths.extend(sorted(folder.glob('manifest.json')))
    paths += list(notebook_dir.glob('08*.ipynb')) + list(notebook_dir.glob('09*.ipynb'))
    paths += [notebook_dir / name for name in
              ('abcd_finalize_common.py', 'abcd_finalize_plots.py')]
    source = notebook_dir / '09_2mu2e_final_check_outputs'
    paths += [source / name for name in ('figure_manifest.csv', 'decision_figure_manifest.csv')]
    return sorted(set(paths))


def finalize_4mu_final_report_diagnostic_candidates(result):
    """Compare an extended loose A with fixed-A buffers; never adopt a failed point."""
    scan = result['tables']['mc_nominal_scan']
    loose_scan = scan.loc[scan.strategy.eq('loose') & finalize_4mu_final_report_np.isfinite(scan.z_known_b_median)].copy()
    if len(loose_scan) != 20:
        raise AssertionError('The loose diagnostic must compare all twenty finite native MC scores.')
    best = loose_scan.sort_values(['z_known_b_median','a_iso','a_phi','candidate_id'],
                                  ascending=[False,True,False,True]).iloc[0]
    rows = [dict(best.to_dict(), label='Loose extended (MC-score diagnostic)', role='diagnostic',
                 group='loose_extended', is_primary=True,
                 diagnostic_basis='MC_SCORE_ONLY_BEFORE_PURITY_GATE')]
    for anchor, isolation in (('WP2', .25), ('WP1', .3)):
        match = scan.loc[scan.strategy.eq('loose') & finalize_4mu_final_report_np.isclose(scan.a_iso,isolation)
                         & finalize_4mu_final_report_np.isclose(scan.a_phi,2.4)]
        assert len(match) == 1
        row = match.iloc[0].to_dict()
        if row['candidate_id'] != best.candidate_id:
            rows.append(dict(row, label=f'Loose {anchor} (reference only)', role='diagnostic',
                             group='loose_'+anchor.lower(), is_primary=False,
                             diagnostic_basis='FIXED_ANCHOR_REFERENCE'))
        buffers = scan.loc[scan.strategy.eq('buffer_'+anchor.lower())
                           & scan.main_scan & scan.positive_BCD
                           & finalize_4mu_final_report_np.isfinite(scan.max_S_B_BCD)].copy()
        if buffers.empty:
            buffers = scan.loc[scan.strategy.eq('buffer_'+anchor.lower()) & scan.main_scan].copy()
        iso_edges = [.25,.3,.4,.5,.75,1.,2.]
        phi_edges = [.5,1.,1.5,2.,2.2,2.4]
        buffers['native_movement'] = [iso_edges.index(float(x))-iso_edges.index(isolation)
                                      +phi_edges.index(2.4)-phi_edges.index(float(y))
                                      for x,y in zip(buffers.b_iso,buffers.c_phi)]
        buffers = buffers.sort_values(['max_S_B_BCD','native_movement','b_iso','c_phi'],
                                       ascending=[True,True,True,False],na_position='last')
        row = buffers.iloc[0].to_dict()
        rows.append(dict(row,label=f'Buffered {anchor}',role='diagnostic',
                         group='buffer_'+anchor.lower(), is_primary=anchor=='WP2',
                         diagnostic_basis='FIXED_A_MINIMUM_SUPPORTED_CONTROL_CONTAMINATION'))
    candidates = finalize_4mu_final_report_pd.DataFrame(rows)
    if candidates.candidate_id.duplicated().any():
        raise AssertionError('Primary and reference diagnostics must not duplicate geometries.')
    if set(candidates.loc[candidates.is_primary,'group']) != {'loose_extended','buffer_wp2'}:
        raise AssertionError('The final comparison needs one extended loose A and one fixed WP2 A.')
    result['config']['loose_diagnostic'] = {
        field: best[field] for field in ('candidate_id','a_iso','a_phi','b_iso','c_phi')}
    result['config']['loose_diagnostic']['ranking'] = 'MC_SCORE_ONLY_BEFORE_PURITY_GATE'
    return candidates


def finalize_4mu_final_report_summary_tables(notebook_dir, result):
    scan = result['tables']['mc_nominal_scan']
    diagnostics = result['plot_candidates']
    status = []
    for strategy in ('loose','buffer_wp2','buffer_wp1'):
        group = scan.loc[scan.strategy.eq(strategy)]
        supported = group.loc[group.positive_BCD & finalize_4mu_final_report_np.isfinite(group.max_S_B_BCD)]
        best = supported.sort_values('max_S_B_BCD').iloc[0] if len(supported) else None
        status.append(dict(strategy=strategy,candidates=len(group),
            main_candidates=int(group.main_scan.sum()),
            positive_control_candidates=int(group.positive_BCD.sum()),
            nominal_BCD_gate_pass=int(group.purity_pass_BCD.sum()),
            minimum_supported_max_S_B=finalize_4mu_final_report_np.nan if best is None else best.max_S_B_BCD,
            limiting_region='' if best is None else best.worst_BCD_region,
            limiting_sample='' if best is None else best.worst_BCD_sample,
            selected=False, status='NO_NOMINAL_GATE_PASS' if not group.purity_pass_BCD.any()
                                  else 'MC_ELIGIBLE_REVIEW_REQUIRED',
            interpretation='Minimum contamination is diagnostic; this is not an adopted point satisfying statistical support and disclosure requirements'))
    regions = result['tables']['mc_regions']
    diagnostic_regions = diagnostics[['candidate_id','group','label']].merge(
        regions.loc[regions.scope.eq('nominal_DD')],on='candidate_id',validate='one_to_many')
    diagnostic_data = diagnostics[['candidate_id','group','label']].merge(
        result['tables']['data_closure'],on='candidate_id',validate='one_to_many')
    c_support = regions.loc[regions.scope.eq('nominal_DD') & regions.region.eq('C')
                           & regions.strategy.str.startswith('buffer_')].copy()
    c_support = c_support.sort_values(['anchor','c_phi','b_iso']).drop_duplicates(['anchor','c_phi'])
    b_support = regions.loc[regions.scope.eq('nominal_DD') & regions.region.eq('B')
                           & regions.strategy.str.startswith('buffer_')].copy()
    b_support = b_support.sort_values(['anchor','b_iso','c_phi']).drop_duplicates(['anchor','b_iso'])
    previous = finalize_4mu_final_report_pd.read_csv(notebook_dir/'08_4mu_VR2_focused_outputs'/'selected_comprehensive_numeric.csv',
                           float_precision='round_trip')
    previous = previous.loc[previous.wp.isin(['WP1','WP2','Ref'])].copy()
    loose_ranking = scan.loc[scan.strategy.eq('loose')].sort_values(
        ['z_known_b_median','a_iso','a_phi','candidate_id'],
        ascending=[False,True,False,True]).copy()
    loose_ranking['MC_score_rank_before_purity_gate'] = finalize_4mu_final_report_np.arange(1,len(loose_ranking)+1)
    loose_identity = diagnostics.loc[diagnostics.group.eq('loose_extended'),'candidate_id'].iloc[0]
    loose_ranking['shown_as_extended_loose_diagnostic'] = loose_ranking.candidate_id.eq(loose_identity)
    loose_ranking['adopted_nominal_method'] = False
    components = diagnostics[['candidate_id','group','label']].merge(
        result['tables']['mc_background_components'].loc[
            result['tables']['mc_background_components'].scope.eq('nominal_DD')],
        on='candidate_id',validate='one_to_many')
    components['error'] = finalize_4mu_final_report_np.sqrt(components['variance'])
    settings = finalize_4mu_final_report_pd.DataFrame([
        ('Channel and ordering','Reconstructed 4μ; LJ0=leading μ-LJ, LJ1=subleading μ-LJ'),
        ('Nominal','DD, M_LJ,LJ≥150 GeV, 0≤I1<0.5'),
        ('Displacement','P: max pixel hits>2; D: ≤2, applied to each μ-LJ; DSA constituent=0'),
        ('Primary validation','150≤M_LJ,LJ<300 GeV, PP/PD; DP is an auxiliary validation with limited statistics'),
        ('Additional scopes','0≤M<150 GeV and inclusive MC; intermediate DD is MC-only'),
        ('Loose main scan','I_A=.20,.25,.30,.40,.50 × phi_A=2.4,2.6,2.7,2.8'),
        ('Loose comparison','Maximum absolute median known-background Z over the primary 20 points before the gate; unadopted diagnostic candidate with extended A'),
        ('Buffer anchors','WP2 A=(.25,2.4) is primary; WP1 A=(.30,2.4) is auxiliary'),
        ('Buffer controls','Native edges with I_H≥I_A; phi_L=.5,1,1.5,2,2.2,2.4; wide I_H=.75,1,2 are separate'),
        ('Signal comparison','Sensitivity: 60 matching 4Mu hypotheses; contamination: 120 separate hypotheses; models are never summed'),
        ('Normalization','Preserve stored luminosity and model-specific cross-section weights; no new rescaling'),
        ('MC statistics','sumw, sumw2 and N_eff=(sumw)^2/sumw2; no separate raw-entry input'),
        ('Flow','Exclude isolation underflow; as in 09, MC/Data display [1,2] represents physical [1,∞), with angle flows folded into end bins; region integrals are preserved'),
        ('Data disclosure','Reuse only existing same-scope public rectangles/row space; zero original Data queries or new information'),
        ('Data 2D','Check each bin MC gate and public reuse separately; do not calculate ABCD closure from partial 2D displays'),
        ('Finalization','Require all-120 nominal B/C/D central S/B<10%, valid background support, VR validation and control statistics'),
    ],columns=['item','definition'])
    return dict(decision_summary=finalize_4mu_final_report_pd.DataFrame(status),diagnostic_nominal_regions=diagnostic_regions,
        diagnostic_data_closure=diagnostic_data,buffer_C_support=c_support,buffer_B_support=b_support,
        diagnostic_background_components=components,baseline_08=previous,selection_definitions=settings,
        final_comparison_candidates=diagnostics.loc[diagnostics.is_primary].copy(),
        loose_diagnostic_ranking=loose_ranking,
        wide_loose_audit=finalize_4mu_final_report_wide_loose_audit(notebook_dir))


def finalize_4mu_final_report_wide_loose_audit(notebook_dir):
    """Record the full existing 154-point MC grid without changing the main scan."""
    directory = notebook_dir/'04_4mu_VR2_summary_outputs'
    regions = finalize_4mu_final_report_pd.read_csv(directory/'sr_reference_purity_grid_regions.csv',float_precision='round_trip')
    grid = finalize_4mu_final_report_pd.read_csv(directory/'sr_reference_grid.csv',float_precision='round_trip')
    rows = []
    for (isolation,angle), group in regions.loc[regions.region.isin(list('BCD'))].groupby(['tx','ty']):
        group = group.set_index('region').loc[list('BCD')]
        valid = finalize_4mu_final_report_np.isfinite(group.max_S_B).all()
        rows.append(dict(tx=isolation,ty=angle,nominal_BCD_gate_pass=bool(group.approved.all()),
            all_control_ratios_defined=bool(valid),
            max_S_B_BCD=float(group.max_S_B.max()) if valid else finalize_4mu_final_report_np.nan))
    return grid.merge(finalize_4mu_final_report_pd.DataFrame(rows),on=['tx','ty'],validate='one_to_one')


def finalize_4mu_final_report_figure_section(name):
    if name.startswith(('loose_',)):
        return 'loose'
    if name.startswith(('buffer_',)):
        return 'buffer'
    if name.startswith(('vr_', 'validation_', 'data2d_', 'mc2d_')):
        return 'validation'
    if name.startswith(('final_', 'comparison_', 'model_', 'expected_')):
        return 'comparison'
    if name.startswith(('nominal_', 'overview_', 'approaches_', 'candidate_', 'signal_')):
        return 'overview'
    return 'backup'


def finalize_4mu_final_report_json_default(value):
    if isinstance(value,finalize_4mu_final_report_Path):
        return str(value)
    if isinstance(value,finalize_4mu_final_report_np.generic):
        return value.item()
    if isinstance(value,finalize_4mu_final_report_np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def finalize_4mu_final_report_save_report(result,notebook_dir):
    reference_presentation_order = _FINALIZE_NAMESPACES['abcd_4mu_final_09style'].reference_presentation_order

    output = result['output_dir']
    tables_dir = output/'tables'
    tables_dir.mkdir(parents=True,exist_ok=True)
    definitions = result['tables'].get('figure_definitions',finalize_4mu_final_report_pd.DataFrame())
    sections = (definitions.set_index('name').section.to_dict()
                if {'name','section'}.issubset(definitions) else {})
    order,decision = reference_presentation_order(notebook_dir)
    figure_map = dict(result['figures'])
    if list(figure_map) != order:
        raise AssertionError('The 4mu presentation must follow the exact notebook-09 order.')
    catalog=[]
    for page,name in enumerate(order,1):
        figure = figure_map[name]
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('Unsafe figure name: '+name)
        section = sections.get(name,finalize_4mu_final_report_figure_section(name))
        section = {'nominal':'overview'}.get(section,section)
        if section not in finalize_4mu_final_report_SECTION_LABELS:
            section = finalize_4mu_final_report_figure_section(name)
        figure.savefig(output/(name+'.png'),dpi=120,bbox_inches='tight',facecolor='white')
        catalog.append(dict(order=page,name=name,section=section,
            caption=result['captions'][name],png=name+'.png'))
    catalog = finalize_4mu_final_report_pd.DataFrame(catalog)
    if len(catalog)!=catalog.name.nunique() or set(catalog.name)!=set(result['captions']):
        raise AssertionError('Each unique figure needs exactly one external caption.')
    if catalog.name.tolist() != order:
        raise AssertionError('Catalog ordering must not regroup notebook-09 presentation slots.')
    result['catalog'] = catalog
    result['tables']['figure_catalog'] = catalog
    result['tables']['figure_manifest'] = catalog[['name','png','order']].copy()
    result['tables']['decision_figure_manifest'] = finalize_4mu_final_report_pd.DataFrame([
        dict(name=name,png=name+'.png',order=page)
        for page,name in enumerate(decision,1)])
    result['tables']['input_manifest'] = finalize_4mu_final_report_pd.concat([
        result['tables']['mc_input_manifest'],result['tables']['data_prior_sources']],ignore_index=True)
    table_files = finalize_4mu_final_report__export_numerical_tables(result['tables'],tables_dir)
    # Match 09's familiar output names as well as its slide order.
    for name in ('figure_manifest','decision_figure_manifest'):
        result['tables'][name].to_csv(output/(name+'.csv'),index=False)
    presentation_sequences=dict(full=order,decision=decision)
    implementations={name:finalize_4mu_final_report_file_hash(notebook_dir/name)
                     for name in ('abcd_finalize_common.py', 'abcd_finalize_plots.py')}
    manifest=dict(created_utc=finalize_4mu_final_report_datetime.now(finalize_4mu_final_report_timezone.utc).isoformat(),
        configuration=result['config'],methodology=result['metadata'],figures=catalog.to_dict('records'),
        presentation_sequences=presentation_sequences,
        table_rows={name:len(table) for name,table in result['tables'].items()},
        table_files=table_files,
        implementation_sha256=implementations,nominal_A_evaluated=False,
        nominal_BCD_evaluated=False,final_method=None,
        final_status=result['metadata']['finalization_status'])
    (output/'manifest.json').write_text(finalize_4mu_final_report_json.dumps(manifest,ensure_ascii=False,indent=2,default=finalize_4mu_final_report_json_default)+'\n')
    primary = result['tables']['final_comparison_candidates'].set_index('group')
    loose_comparison = primary.loc['loose_extended']
    buffer_comparison = primary.loc['buffer_wp2']
    lines=['4mu loose boundary / fixed-A buffer study','',
        'Scope: reconstructed 4mu only; unchanged weights and 120 separate hypotheses.',
        f'Loose extends the common A/B/C/D boundaries: I0<{loose_comparison.a_iso:g}, |dphi|>={loose_comparison.a_phi:g}.',
        'The loose illustration maximizes absolute median known-background MC-A Z among the 20 scan points BEFORE the nominal purity gate. It is an ineligible diagnostic, not an adopted optimum.',
        f'Buffer keeps WP2 A fixed: I0<{buffer_comparison.a_iso:g}, |dphi|>={buffer_comparison.a_phi:g}; B/D start at {buffer_comparison.b_iso:g}, C/D end at {buffer_comparison.c_phi:g}.',
        'The A definitions differ. Buffer-scan MC-A sensitivity remains constant within that approach; it is not constrained to equal the extended-loose sensitivity.',
        'At these diagnostic definitions buffer B overlaps part of the extended loose A. The existing reviewed-A protection remains in force; no nominal control query is performed.',
        'No nominal DATA A or B/C/D is opened. Public VR counts are reused from the same selection only.',
        'Current finalization status: '+result['metadata']['finalization_status'],'']
    for row in result['tables']['decision_summary'].itertuples():
        lines.append(f'{row.strategy}: {row.nominal_BCD_gate_pass}/{row.candidates} pass; '
                     f'minimum supported max(S/B)={row.minimum_supported_max_S_B:.6g}.')
    lines += ['', 'The illustrated points are diagnostics, not adopted nominal definitions.',
        'Zero or sparse MC support is not a measurement of zero physical background.',
        'VR closure does not by itself authorize nominal controls. Nonclosure transfer and a full likelihood remain open.']
    (output/'final_team_proposal.txt').write_text('\n'.join(lines)+'\n')


def finalize_4mu_final_report_show_4mu_section(report,section):
    from IPython.display import Markdown as Markdown; from IPython.display import display as display
    figures=dict(report['figures'])
    questions=report['tables']['figure_definitions'].set_index('name').question.to_dict()
    for row in report['catalog'].loc[report['catalog'].section.eq(section)].itertuples():
        display(Markdown(f'#### Figure {row.order:02d} · {questions[row.name]}\n\n'+row.caption))
        display(figures[row.name])


def finalize_4mu_final_report__export_numerical_tables(tables,tables_dir):
    """Keep the large signal audit compressed, with explicit manifest paths."""
    files = {}
    for name,table in tables.items():
        if not isinstance(table,finalize_4mu_final_report_pd.DataFrame):
            raise TypeError('Expected a numerical table: '+name)
        compressed = name == 'mc_signals'
        filename = name + ('.csv.gz' if compressed else '.csv')
        compression = {'method':'gzip','compresslevel':6,'mtime':0} if compressed else None
        table.to_csv(tables_dir/filename,index=False,compression=compression)
        if compressed:
            (tables_dir/(name+'.csv')).unlink(missing_ok=True)
        files[name] = dict(path='tables/'+filename,format='csv',
                          compression='gzip' if compressed else None)
    return files


# ========================================================================
# abcd_finalize_common: original study API
# ========================================================================
"""Shared ABCD_finalize calculations using the histogram/statistics conventions of notebooks 13–16."""
from pathlib import Path as Path
import gc
import hashlib
import json
import sys
from datetime import datetime as datetime; from datetime import timezone as timezone

import coffea.util
import numpy as np
import pandas as pd


# Shared numeric binning and flow folding; independent of plotting imports.
def _opt_distribution_edges(planes, key, upper):
    """Use only common native edges, with the same exact endpoints as 05–10."""
    arrays = [np.asarray(plane[key], dtype=float) for plane in planes if plane is not None]
    if not arrays:
        return np.array([0., float(upper)])
    candidates = arrays[0]
    shared = np.array([edge for edge in candidates
                       if edge >= -1e-10 and edge <= upper + 1e-10
                       and all(np.any(np.isclose(other, edge, rtol=0, atol=1e-9)) for other in arrays[1:])])
    if len(shared) < 2 or not np.isclose(shared[-1], upper, rtol=0, atol=1e-9):
        raise ValueError(f"A common exact display binning ending at {upper:g} is required for {key}.")
    if not all(np.isclose(shared[0], array[0], rtol=0, atol=1e-9) for array in arrays):
        raise ValueError("Histogram lower edges differ; display folding cannot discard physical bins.")
    return shared

def _opt_fold_distribution(plane, xedges, yedges, angular, normalize=False):
    """Apply the original display-only folding, retaining each allowed flow once."""
    values = np.asarray(plane["values"], dtype=float)
    native_x = np.asarray(plane["xedges"], dtype=float)
    native_y = np.asarray(plane["yedges"], dtype=float)
    if values.shape != (len(native_x) + 1, len(native_y) + 1):
        raise ValueError("A plane must include numeric flow bins on both axes.")
    selected = values[1:, :] if angular else values[1:, 1:]
    total = float(selected.sum())
    if normalize and (not np.isfinite(selected).all() or not np.isfinite(total) or total <= 0):
        return None, total
    def indices(native, common):
        lower = native[:-1].copy()
        for edge in common:
            if not np.any(np.isclose(native, edge, rtol=0, atol=1e-9)):
                raise ValueError("Display boundaries must be exact native edges.")
            lower[np.isclose(lower, edge, rtol=0, atol=1e-9)] = edge
        regular = np.clip(np.searchsorted(common, lower, side="right") - 1, 0, len(common) - 2)
        return np.r_[regular, len(common) - 2]
    ix, iy = indices(native_x, xedges), indices(native_y, yedges)
    if angular:
        iy = np.r_[0, iy]
    xx, yy = np.meshgrid(ix, iy, indexing="ij")
    folded = np.zeros((len(xedges) - 1, len(yedges) - 1), dtype=float)
    np.add.at(folded, (xx.ravel(), yy.ravel()), selected.ravel())
    return (folded / total if normalize else folded), total

if int(np.__version__.split(".")[0]) < 2:
    import numpy.core as _numpy_core
    import numpy.core.multiarray as _numpy_multiarray
    import numpy.core.numeric as _numpy_numeric
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", _numpy_numeric)

# Retain only the required histogram schema, statistics, and flow-aware helpers.
COMPARISON_VARIABLES = ("iso0", "iso1", "disp0", "disp1", "abs_dphi", "ljlj_mass")

def _comparison_json(value):
    def convert(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item, key=str)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        return str(item)
    return json.dumps(value, sort_keys=True, default=convert)

def _comparison_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _comparison_group(sample):
    if sample.startswith("QCD_"):
        return "QCD"
    if sample.startswith("DYJetsTo"):
        return "DY"
    if sample == "TTJets":
        return "TTJets"
    if sample in {"WW", "WZ", "ZZ"}:
        return "Diboson"
    raise ValueError(f"Unexpected MC background sample: {sample}")

def _comparison_split(edges, threshold):
    matches = np.flatnonzero(np.isclose(edges, float(threshold), rtol=0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError(f"A requested threshold is not a unique stored bin edge: {threshold}")
    return int(matches[0]) + 1

def _comparison_validate_arrays(values, variances):
    if variances is None or values.shape != variances.shape:
        raise ValueError("Weighted values and sumw2 must have the same shape.")
    if not np.isfinite(values).all() or not np.isfinite(variances).all():
        raise ValueError("The selected histogram contains nonfinite weights or sumw2.")
    if np.any(variances < 0):
        raise ValueError("The selected histogram contains negative sumw2.")

def _comparison_plane(values, variances, xedges, yedges):
    values = np.asarray(values, dtype=float).copy()
    variances = np.asarray(variances, dtype=float).copy()
    _comparison_validate_arrays(values, variances)
    xedges = np.asarray(xedges, dtype=float).copy()
    yedges = np.asarray(yedges, dtype=float).copy()
    if values.shape != (len(xedges) + 1, len(yedges) + 1):
        raise ValueError("The 2D plane must contain both numeric flow bins on each axis.")
    return dict(values=values, variances=variances, xedges=xedges, yedges=yedges)

def _comparison_add(target, incoming, component):
    if target is None:
        target = _comparison_plane(incoming["values"], incoming["variances"],
                                   incoming["xedges"], incoming["yedges"])
        target["values"][:] = 0
        target["variances"][:] = 0
        target["components"] = {}
    if not all(np.array_equal(target[key], incoming[key]) for key in ("xedges", "yedges")):
        raise ValueError("Cannot combine planes with different numeric bin edges.")
    target["values"] += incoming["values"]
    target["variances"] += incoming["variances"]
    if component not in target["components"]:
        target["components"][component] = dict(
            values=np.zeros_like(incoming["values"]),
            variances=np.zeros_like(incoming["variances"]))
    target["components"][component]["values"] += incoming["values"]
    target["components"][component]["variances"] += incoming["variances"]
    return target

def _comparison_payload(path, source):
    loaded = coffea.util.load(path)
    outputs = loaded.get("out", loaded) if isinstance(loaded, dict) else None
    if outputs is None or set(outputs) != {path.stem}:
        raise ValueError(f"The payload sample identity does not match its file name: {path}")
    payload = outputs[path.stem]
    metadata = payload.get("metadata", {})
    if set(metadata.get("is_data", ())) != {False}:
        raise ValueError(f"Only explicitly identified MC backgrounds may be loaded: {path}")
    if set(metadata.get("unweighted_hist", ())) != {False}:
        raise ValueError(f"A weighted production is required: {path}")
    if {str(value) for value in metadata.get("year", ())} != {"2018"}:
        raise ValueError(f"The payload must have a unique 2018 year: {path}")
    if "hists" not in payload or "cutflow" not in payload:
        raise ValueError(f"Missing histograms or production cutflows: {path}")
    return loaded, payload

def _comparison_validate_hist(histogram, path):
    if histogram.storage_type.__name__ != "Weight":
        raise ValueError(f"Weight storage is required: {path}")
    for axis in histogram.axes:
        if not axis.traits.discrete and not (axis.traits.underflow and axis.traits.overflow):
            raise ValueError(f"Numeric flow bins are required: {path}/{axis.name}")

def _comparison_current_cube(payload, path, config):
    expected_hists = {f"abcd6d_{topology}" for topology in ("2mu2e", "4mu")} | {
        f"abcd6d_ref_{topology}_{pair}" for topology in ("2mu2e", "4mu")
        for pair in ("iso0_vs_iso1", "iso0_vs_abs_dphi", "disp0_vs_disp1")}
    if set(payload["hists"]) != expected_hists:
        raise ValueError(f"The current 6D campaign histogram schema differs: {path}")
    histogram = payload["hists"][config["current_hist"]]
    _comparison_validate_hist(histogram, path)
    if tuple(axis.name for axis in histogram.axes) != ("channel",) + COMPARISON_VARIABLES:
        raise ValueError(f"Current 6D axis order mismatch: {path}")
    channel_axis = histogram.axes["channel"]
    if (list(channel_axis) != [config["expected_current_channel"]]
            or not channel_axis.traits.discrete
            or channel_axis.traits.underflow or channel_axis.traits.overflow):
        raise ValueError(f"Current topology channel identity/flow mismatch: {path}")
    expected_edges = {
        "iso0": [0, .05, .1, .15, .2, .25, .3, .4, .5, .75, 1, 2, 5],
        "iso1": [0, .05, .1, .15, .2, .25, .3, .4, .5, .75, 1, 2, 5],
        "disp0": np.arange(-.5, 12, 1),
        "disp1": np.arange(-.5, 10 if config["topology"] == "2mu2e" else 12, 1),
        "abs_dphi": [0, .5, 1, 1.5, 2, 2.2, 2.4, 2.6, 2.7, 2.8, 2.9, 3, 3.05, 3.1, np.pi],
        "ljlj_mass": [0, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2500],
    }
    edges = {name: np.asarray(histogram.axes[name].edges, dtype=float) for name in COMPARISON_VARIABLES}
    for name in COMPARISON_VARIABLES:
        expected = np.asarray(expected_edges[name], dtype=float)
        if edges[name].shape != expected.shape or not np.allclose(edges[name], expected, rtol=0, atol=1e-12):
            raise ValueError(f"Current campaign stored bin edges differ: {path}/{name}")
    weights, variances = histogram.values(flow=True), histogram.variances(flow=True)
    expected_shape = (1,) + tuple(len(edges[name]) + 1 for name in COMPARISON_VARIABLES)
    if variances is None or weights.shape != expected_shape or variances.shape != expected_shape:
        raise ValueError(f"Current 6D weighted values/variance shape mismatch: {path}")
    return weights[0], variances[0], edges

def plane_statistics(plane, tx, ty, include_iso_underflow=False, min_neff=10.):
    """Match notebook 10 prediction/closure conventions; annotate edge cases.

    B, C and D are total background sums. No negative sumw is clipped. The
    relative-error formula intentionally reproduces notebook 10's zero-sideband
    uncertainty convention; its mathematical limitation is flagged in status.
    """
    regions = _comparison_regions(plane, tx, ty, include_iso_underflow)
    row = dict(tx=float(tx), ty=float(ty), pred=np.nan, pred_error=np.nan,
               kappa=np.nan, kappa_error=np.nan, pull=np.nan,
               status="UNDEFINED_PREDICTION", kappa_status="UNDEFINED_PREDICTION")
    for name, item in regions.items():
        row.update({name: item["value"], "var_" + name: item["variance"],
                    name + "_error": item["error"], "neff_" + name: item["neff"]})
    neffs = np.array([regions[name]["neff"] for name in "ABCD"])
    row.update(neff_min=float(np.min(neffs)), min_neff=float(np.min(neffs)),
               min_sideband_neff=float(np.min(neffs[1:])),
               low_stat=bool(not np.all(np.isfinite(neffs) & (neffs >= min_neff))))
    a, b, c, d = (regions[name]["value"] for name in "ABCD")
    ea, eb, ec, ed = (regions[name]["error"] for name in "ABCD")
    if d <= 0 or not np.all(np.isfinite([b, eb, c, ec, d, ed])):
        row["status"] = "NONPOSITIVE_D_OR_NONFINITE_SIDEBAND"
        row["kappa_status"] = row["status"]
        return row
    pred = b * c / d
    terms = [(eb / b)**2 if b else 0., (ec / c)**2 if c else 0., (ed / d)**2]
    epred = abs(pred) * np.sqrt(sum(terms))
    row.update(pred=pred, pred_error=epred)
    if b == 0 or c == 0:
        row["status"] = "ZERO_SIDEBAND_LEGACY_ZERO_PREDICTION_ERROR"
    elif b < 0 or c < 0:
        row["status"] = "NEGATIVE_SIDEBAND_RETAINED"
    else:
        row["status"] = "LOW_STAT_SIDEBAND" if not np.all(
            np.isfinite(neffs[1:]) & (neffs[1:] >= min_neff)) else "OK"
    if np.isfinite(a) and pred > 0:
        row["kappa"] = a / pred
        if a > 0:
            row["kappa_error"] = abs(row["kappa"]) * np.sqrt((ea / a)**2 + (epred / pred)**2)
            row["kappa_status"] = "LOW_STAT" if row["low_stat"] else "OK"
        else:
            row["kappa_status"] = "NONPOSITIVE_A_LEGACY_KAPPA_ERROR_UNDEFINED"
    else:
        row["kappa_status"] = "NONPOSITIVE_PREDICTION_OR_NONFINITE_A"
    if np.isfinite(a) and ea**2 + epred**2 > 0:
        row["pull"] = (a - pred) / np.sqrt(ea**2 + epred**2)
    return row

def _extension_coarsen_regular(plane, ex, ey, key):
    """Exclude flow and regular bins outside the common support; never fold tails."""
    native_x, native_y = plane["xedges"], plane["yedges"]
    values = np.asarray(plane[key], dtype=float)[1:-1, 1:-1]
    selected_x = (native_x[:-1] >= ex[0] - 1e-12) & (native_x[1:] <= ex[-1] + 1e-12)
    selected_y = (native_y[:-1] >= ey[0] - 1e-12) & (native_y[1:] <= ey[-1] + 1e-12)
    low_x, low_y = native_x[:-1][selected_x].copy(), native_y[:-1][selected_y].copy()
    for edge in ex:
        low_x[np.isclose(low_x, edge, rtol=0, atol=1e-12)] = edge
    for edge in ey:
        low_y[np.isclose(low_y, edge, rtol=0, atol=1e-12)] = edge
    ix, iy = np.searchsorted(ex, low_x, side="right") - 1, np.searchsorted(ey, low_y, side="right") - 1
    xx, yy = np.meshgrid(ix, iy, indexing="ij")
    output = np.zeros((len(ex) - 1, len(ey) - 1), dtype=float)
    np.add.at(output, (xx.ravel(), yy.ravel()), values[np.ix_(selected_x, selected_y)].ravel())
    return output

def extension_correlations(plane, tx, ty, ex, ey, min_neff=10.):
    """Weighted Pearson in All/A/B/C/D at centers of shared coarsened bins."""
    regions = ("All", "A", "B", "C", "D")
    if plane is None:
        return [dict(region=region, correlation=np.nan, rho=np.nan, neff=np.nan,
                     status="GATE_BLOCKED", sumw=np.nan, sumw2=np.nan, negative_bins=np.nan,
                     display_binning=_comparison_json(dict(xedges=ex, yedges=ey)),
                     support="Native regular-bin support only; both plane flow axes excluded")
                for region in regions]
    values = _extension_coarsen_regular(plane, ex, ey, "values")
    variances = _extension_coarsen_regular(plane, ex, ey, "variances")
    negative_plane = dict(plane, values=(plane["values"] < 0).astype(float))
    native_negative = _extension_coarsen_regular(negative_plane, ex, ey, "values")
    xx, yy = np.broadcast_arrays(((ex[:-1] + ex[1:]) / 2)[:, None], ((ey[:-1] + ey[1:]) / 2)[None, :])
    xpass, ypass = xx < tx, yy >= ty
    masks = (np.ones(xx.shape, bool), xpass & ypass, ~xpass & ypass, xpass & ~ypass, ~xpass & ~ypass)
    rows = []
    for region, mask in zip(regions, masks):
        weights, vars_ = values[mask], variances[mask]
        row = dict(region=region, correlation=np.nan, rho=np.nan, neff=np.nan, status="EMPTY",
                   sumw=float(weights.sum()), sumw2=float(vars_.sum()), negative_bins=int(np.sum(weights < 0)),
                   native_negative_bins=int(native_negative[mask].sum()),
                   display_binning=_comparison_json(dict(xedges=ex, yedges=ey)),
                   support="Native regular-bin support only; both plane flow axes excluded")
        if not np.isfinite(weights).all() or not np.isfinite(vars_).all() or np.any(vars_ < 0):
            row["status"] = "INVALID_WEIGHT_OR_VARIANCE"
        elif np.any(weights < 0) or row["native_negative_bins"] > 0:
            row["status"] = "NEGATIVE_BINS_NOT_A_PROBABILITY_DISTRIBUTION"
        elif row["sumw"] > 0 and row["sumw2"] > 0:
            row["neff"] = row["sumw"]**2 / row["sumw2"]
            xs, ys = xx[mask], yy[mask]
            mx, my = np.dot(weights, xs) / row["sumw"], np.dot(weights, ys) / row["sumw"]
            vx = np.dot(weights, (xs - mx)**2) / row["sumw"]
            vy = np.dot(weights, (ys - my)**2) / row["sumw"]
            if vx <= 0 or vy <= 0:
                row["status"] = "NO_SPREAD"
            else:
                rho = float(np.dot(weights, (xs - mx) * (ys - my)) / row["sumw"] / np.sqrt(vx * vy))
                if not np.isfinite(rho) or abs(rho) > 1 + 1e-10:
                    row["status"] = "INVALID_MOMENTS"
                else:
                    row.update(rho=float(np.clip(rho, -1, 1)), correlation=float(np.clip(rho, -1, 1)),
                               status="LOW_STAT" if row["neff"] < min_neff else "OK")
        elif row["sumw"] > 0:
            row["status"] = "ZERO_VARIANCE_UNSUPPORTED"
        rows.append(row)
    return rows

def _extension_data_payload(path):
    """Validate DATA identity without inspecting any DATA SR selection or values."""
    loaded = coffea.util.load(path)
    outputs = loaded.get("out", loaded) if isinstance(loaded, dict) else None
    if outputs is None or set(outputs) != {path.stem}:
        raise ValueError(f"DATA payload sample identity mismatch: {path}")
    payload = outputs[path.stem]
    metadata = payload.get("metadata", {})
    if set(metadata.get("is_data", [])) != {True}:
        raise ValueError(f"Explicit DATA metadata are required: {path}")
    if set(metadata.get("unweighted_hist", [])) != {False}:
        raise ValueError(f"Expected weighted DATA storage configuration: {path}")
    if {str(value) for value in metadata.get("year", [])} != {"2018"}:
        raise ValueError(f"Expected unique 2018 DATA metadata: {path}")
    return loaded, payload

def _extension_blank_statistics(tx, ty, reason="GATE_BLOCKED"):
    row = dict(tx=float(tx), ty=float(ty), pred=np.nan, pred_error=np.nan,
               kappa=np.nan, kappa_error=np.nan, pull=np.nan, neff_min=np.nan, min_neff=np.nan,
               min_sideband_neff=np.nan, low_stat=True, status=reason, kappa_status=reason)
    for region in "ABCD":
        row.update({region: np.nan, "var_" + region: np.nan, region + "_error": np.nan, "neff_" + region: np.nan})
    return row

def _angle_regions(plane, tx, ty, include_iso_underflow=False):
    """A=low isolation/high angle; retain both angle flows and exclude iso UF."""
    if include_iso_underflow:
        raise ValueError("This plane comparison excludes isolation underflow in every ABCD region.")
    ix = _comparison_split(plane["xedges"], tx)
    iy = _comparison_split(plane["yedges"], ty)
    slices = {"A": (slice(1, ix), slice(iy, None)),
              "B": (slice(ix, None), slice(iy, None)),
              "C": (slice(1, ix), slice(None, iy)),
              "D": (slice(ix, None), slice(None, iy))}
    result = {}
    for region, index in slices.items():
        value = float(plane["values"][index].sum())
        variance = float(plane["variances"][index].sum())
        result[region] = dict(value=value, variance=variance, error=np.sqrt(variance),
                              neff=value**2 / variance if variance > 0 else np.nan)
    return result

def _comparison_regions(plane, tx, ty, include_iso_underflow=False):
    return _angle_regions(plane, tx, ty, include_iso_underflow)

# Recalculate the explicitly specified VRs from the original DATA, MC, and full signal set.
def candidate_specs(config):
    validate_finalize_config(config)
    specs = {}
    mass = f"{config['mass_min']:g} ≤ M < {config['mass_max']:g} GeV"
    for key in config['candidate_order']:
        specs[key] = dict(
            state0=key[0], state1=key[1],
            partner_iso_low=0., partner_iso_high=.5,
            mass_low=config['mass_min'], mass_high=config['mass_max'],
            tx=config['tx'], ty=config['ty'],
            scan_mu_iso=list(config['scan_mu_iso']), scan_abs_dphi=list(config['scan_abs_dphi']),
            plateau_mu_iso=[], plateau_abs_dphi=[],
            kind='mu_dphi', label=f"{config['vr']} {key}: {key[0]} / {key[1]}",
            selection=f'0 ≤ partner iso < 0.5; {mass}',
            x_label='Mu-LJ isolation' if config['topology'] == '2mu2e' else 'Leading mu-LJ isolation',
            y_label='|Δφ| [rad]')
    document = dict(vr=config['vr'], definitions=specs, signal_hypotheses=120,
                    signal_mass_range_GeV=[200, 1000], leakage_veto=False,
                    selection_origin='User-defined disjoint displacement regions; no DATA optimization')
    return specs, document


def candidate_indices(edges, spec, config):
    validate_vr_spec(spec, config)
    bounds = []
    for index in (0, 1):
        state = spec[f'state{index}']
        split = _comparison_split(edges[f'disp{index}'], .5 if index == 1 and config['topology'] == '2mu2e' else 2.5)
        prompt_low = index == 1 and config['topology'] == '2mu2e'
        if state in ('P', 'D'):
            bound = slice(None, split) if (state == 'P') == prompt_low else slice(split, None)
        else:
            raise ValueError(f'Unknown displacement state: {state}')
        bounds.append(bound)
    low = _comparison_split(edges['iso1'], spec['partner_iso_low'])
    high = None if spec['partner_iso_high'] is None else _comparison_split(edges['iso1'], spec['partner_iso_high'])
    mass_low = _comparison_split(edges['ljlj_mass'], spec['mass_low'])
    mass_high = None if spec['mass_high'] is None else _comparison_split(edges['ljlj_mass'], spec['mass_high'])
    disjoint = (spec['mass_high'] <= 150. or 'P' in (spec['state0'], spec['state1']))
    if not disjoint:
        raise ValueError('Auxiliary selections overlapping the nominal DD SR cannot be projected.')
    return (slice(None), slice(low, high), *bounds, slice(None), slice(mass_low, mass_high))


def project_candidate(weights, variances, edges, spec, config):
    index = candidate_indices(edges, spec, config)
    selected = [array[index] for array in (weights, variances)]
    _comparison_validate_arrays(*selected)
    reduced = [array.sum(axis=(1, 2, 3, 5)) for array in selected]
    for original, plane in zip(selected, reduced):
        np.testing.assert_allclose(plane.sum(), original.sum(), rtol=1e-10, atol=1e-8,
                                   err_msg='Candidate projection failed to conserve sumw/sumw2')
    return dict(_comparison_plane(*reduced, edges['iso0'], edges['abs_dphi']), kind='mu_dphi')


def add_candidate(target, incoming, component):
    return dict(_comparison_add(target, incoming, component), kind='mu_dphi')


def candidate_input_record(path, source):
    return dict(source=source, path=str(path), size_bytes=path.stat().st_size, sha256=_comparison_hash(path))


def candidate_signal_paths(config):
    paths = []
    for path in sorted(config['signal_dir'].glob('*.coffea')):
        fields = path.stem.split('_')
        if len(fields) != 4 or fields[0] not in ('2Mu2E', '4Mu'):
            continue
        mass = float(fields[1].removesuffix('GeV').replace('p', '.'))
        if 200 <= mass <= 1000:
            paths.append(path)
    if len(paths) != 120 or any(sum(p.stem.startswith(decay + '_') for p in paths) != 60
                                for decay in ('2Mu2E', '4Mu')):
        raise ValueError('Expected 120 signal hypotheses across both decay topologies at 200–1000 GeV.')
    return paths


def load_candidate_mc(specs, config):
    planes = {key: {'VR': None} for key in specs}
    signals = {key: {} for key in specs}
    sr = dict(spec=mc_sr_spec(config), plane=None, signals={})
    manifest = []
    backgrounds = sorted(config['bkg_dir'].glob('*.coffea'))
    if len(backgrounds) != 18:
        raise ValueError('Expected 18 background MC samples.')
    signal_paths = candidate_signal_paths(config)
    jobs = [(path, 'Background') for path in backgrounds] + [(path, 'Signal hypothesis') for path in signal_paths]
    for index, (path, source) in enumerate(jobs, 1):
        loaded, payload = _comparison_payload(path, 'sixd')
        weights, variances, edges = _comparison_current_cube(payload, path, config)
        for key, spec in specs.items():
            plane = project_candidate(weights, variances, edges, spec, config)
            if source == 'Background':
                planes[key]['VR'] = add_candidate(planes[key]['VR'], plane, _comparison_group(path.stem))
            else:
                signals[key][path.stem] = plane
        sr_plane = project_mc_sr(weights, variances, edges, config)
        if source == 'Background':
            sr['plane'] = add_candidate(sr['plane'], sr_plane, _comparison_group(path.stem))
        else:
            sr['signals'][path.stem] = sr_plane
        manifest.append(candidate_input_record(path, source))
        del loaded, payload, weights, variances
        gc.collect()
        if index % 20 == 0 or index == len(jobs):
            print(f'MC {index}/{len(jobs)} (18 backgrounds + 120 individual signal hypotheses)', flush=True)
    return planes, signals, manifest, sr


def candidate_coordinates(spec, plane):
    # Scan native boundaries over the full display range; do not reselect WPs using DATA.
    xs = [float(x) for x in spec['scan_mu_iso']]
    ys = [float(y) for y in spec['scan_abs_dphi']]
    for axis, coordinates in [('xedges', xs), ('yedges', ys)]:
        for coordinate in coordinates:
            _comparison_split(plane[axis], coordinate)
    if spec['tx'] not in xs or spec['ty'] not in ys:
        raise ValueError('Stored WP must be a native diagnostic threshold.')
    grid = [dict(tx=x, ty=y) for x in xs for y in ys]
    scan = ([dict(scan_axis='x', threshold=x, tx=x, ty=spec['ty']) for x in xs]
            + [dict(scan_axis='y', threshold=y, tx=spec['tx'], ty=y) for y in ys])
    return scan, grid


def candidate_purity(planes, signals, specs, config, context='VR', require_nominal=False):
    gates, signal_rows, region_rows = {}, [], []
    for key, spec in specs.items():
        for point in candidate_coordinates(spec, planes[key][context])[1]:
            tx, ty = point['tx'], point['ty']
            bkg = _comparison_regions(planes[key][context], tx, ty)
            valid_background = all(np.isfinite(bkg[r]['value']) and bkg[r]['value'] > 1e-8 for r in 'ABCD')
            entries = []
            invalid_signal = False
            for sample, signal in signals[key].items():
                regions = _comparison_regions(signal, tx, ty)
                row = dict(plane=key, context=context, sample=sample, tx=tx, ty=ty)
                for region in 'ABCD':
                    s, vs = regions[region]['value'], regions[region]['variance']
                    b, vb = bkg[region]['value'], bkg[region]['variance']
                    valid_signal = np.isfinite(s) and s >= 0
                    invalid_signal |= not valid_signal
                    ratio = s / b if valid_signal and b > 1e-8 else np.nan
                    error = np.sqrt(vs / b**2 + s*s*vb / b**4) if np.isfinite(ratio) else np.nan
                    row.update({f'S_{region}': s, f'var_S_{region}': vs,
                                f'S_B_{region}': ratio, f'S_B_error_{region}': error})
                    entries.append(dict(sample=sample, region=region, ratio=ratio, error=error))
                signal_rows.append(row)
            finite = [entry for entry in entries if np.isfinite(entry['ratio'])]
            worst = max(finite, key=lambda entry: entry['ratio']) if finite else dict(ratio=np.nan, error=np.nan, sample='', region='')
            approved = valid_background and not invalid_signal and len(finite) == 480 and worst['ratio'] < config['purity_limit']
            gate = dict(approved=bool(approved), purity_pass=bool(approved), max_S_B=worst['ratio'],
                        max_S_B_error=worst['error'], worst_sample=worst['sample'], worst_region=worst['region'],
                        signal_hypotheses=120, valid_background=valid_background,
                        gate_reason='' if approved else 'S/B >= 0.10 or invalid/nonpositive MC support')
            gates[key, tx, ty] = gate
            for region in 'ABCD':
                choices = [entry for entry in finite if entry['region'] == region]
                worst_region = max(choices, key=lambda entry: entry['ratio']) if choices else dict(ratio=np.nan, error=np.nan, sample='')
                region_rows.append(dict(plane=key, context=context, tx=tx, ty=ty, region=region, max_S_B=worst_region['ratio'],
                                        max_S_B_error=worst_region['error'], worst_sample=worst_region['sample'],
                                        approved=bool(len(choices) == 120 and worst_region['ratio'] < config['purity_limit'])))
        if require_nominal and not gates[key, spec['tx'], spec['ty']]['approved']:
            raise ValueError(f'{key}: failed to reproduce the full-grid S/B requirement for the saved nominal candidate.')
    return gates, pd.DataFrame(signal_rows), pd.DataFrame(region_rows)


def load_candidate_data(specs, config, gates):
    planes, manifest = {key: None for key in specs}, []
    paths = sorted(config['data_dir'].glob('*.coffea'))
    if len(paths) != 68:
        raise ValueError('Expected 68 DATA files.')
    allowed = {key: spec for key, spec in specs.items()
               if any(gate['approved'] for (name, _, _), gate in gates.items() if name == key)}
    if not allowed:
        print('All WPs fail the S/B requirement: skipping DATA projection', flush=True)
        return planes, manifest
    for index, path in enumerate(paths, 1):
        loaded, payload = _extension_data_payload(path)
        weights, variances, edges = _comparison_current_cube(payload, path, config)
        for key, spec in allowed.items():
            plane = project_candidate(weights, variances, edges, spec, config)
            planes[key] = add_candidate(planes[key], plane, 'Total')
        manifest.append(candidate_input_record(path, 'DATA selected VR candidates'))
        del loaded, payload, weights, variances
        gc.collect()
        if index % 20 == 0 or index == len(paths):
            print(f'DATA {index}/{len(paths)}', flush=True)
    return planes, manifest


def collect_candidate_tables(planes, data_planes, gates, specs, config):
    closure, scan, grid, composition, correlations = [], [], [], [], []
    for key, spec in specs.items():
        points = {}
        for source, plane in [('MC', planes[key]['VR']), ('DATA', data_planes[key])]:
            for point in candidate_coordinates(spec, planes[key]['VR'])[1]:
                tx, ty = point['tx'], point['ty']
                gate = gates[key, tx, ty]
                stats = (plane_statistics(plane, tx, ty, min_neff=config['min_neff'])
                         if source == 'MC' or gate['approved'] else _extension_blank_statistics(tx, ty))
                passed = (source == 'DATA' and gate['approved'] and .9 <= stats['kappa'] <= 1.1
                          and stats['kappa_error'] <= .2 and min(stats[r] for r in 'ABCD') >= 10)
                row = dict(plane=key, context='VR', source=source, **stats, **gate,
                           in_plateau=False, point_pass=bool(passed))
                grid.append(row)
                points[source, tx, ty] = row
            closure.append(dict(points[source, spec['tx'], spec['ty']]))
            for point in candidate_coordinates(spec, planes[key]['VR'])[0]:
                row = dict(points[source, point['tx'], point['ty']])
                row.update(scan_axis=point['scan_axis'], threshold=point['threshold'])
                scan.append(row)
            nominal_allowed = source == 'MC' or gates[key, spec['tx'], spec['ty']]['approved']
            reference = planes[key]['VR']
            for row in extension_correlations(plane if nominal_allowed else None, spec['tx'], spec['ty'],
                                               reference['xedges'], reference['yedges'], config['min_neff']):
                correlations.append(dict(plane=key, source=source, context='VR', **row))
        background = planes[key]['VR']
        for component, values in background['components'].items():
            for region, entry in _comparison_regions(dict(background, **values), spec['tx'], spec['ty']).items():
                composition.append(dict(plane=key, context='VR', component=component, region=region,
                                        **{'yield': entry['value']}, error=entry['error'], neff=entry['neff']))
    return dict(closure=pd.DataFrame(closure), scan=pd.DataFrame(scan), grid=pd.DataFrame(grid),
                composition=pd.DataFrame(composition), correlations=pd.DataFrame(correlations))


def validate_candidate_tables(tables, specs, config):
    """Verify new selections and gate masking, without requiring good closure."""
    records = []
    for key, spec in specs.items():
        validate_vr_spec(spec, config)
        data = tables['grid'].loc[tables['grid'].plane.eq(key) & tables['grid'].source.eq('DATA')]
        expected = len(spec['scan_mu_iso']) * len(spec['scan_abs_dphi'])
        if len(data) != expected or data.duplicated(['tx', 'ty']).any():
            raise AssertionError(f'{key}: duplicate or missing DATA scan coordinates')
        blocked = data.loc[~data.approved]
        metrics = list('ABCD') + ['var_' + r for r in 'ABCD'] + ['pred', 'pred_error', 'kappa', 'kappa_error']
        if not blocked[metrics].isna().all().all():
            raise AssertionError(f'{key}: DATA values failing the S/B requirement were exposed.')
        nominal = tables['closure'].loc[tables['closure'].plane.eq(key) & tables['closure'].source.eq('DATA')].iloc[0]
        corr = tables['correlations'].loc[tables['correlations'].plane.eq(key) & tables['correlations'].source.eq('DATA')]
        if not nominal.approved and not corr[['rho', 'neff', 'sumw', 'sumw2']].isna().all().all():
            raise AssertionError(f'{key}: blocked DATA correlation values were exposed.')
        records.append(dict(plane=key, vr=config['vr'], mass_low=spec['mass_low'], mass_high=spec['mass_high'],
                            states=spec['state0'] + spec['state1'], scan_points=expected,
                            data_approved_points=int(data.approved.sum()),
                            check='VR whitelist, SR disjointness, unique scan and DATA gate masking', passed=True))
    return pd.DataFrame(records)


def candidate_summary(tables, specs):
    rows = []
    for key, spec in specs.items():
        fixed = tables['closure'].loc[tables['closure'].plane.eq(key)].set_index('source')
        grid = tables['grid'].loc[tables['grid'].plane.eq(key)]
        data, mc = grid.loc[grid.source.eq('DATA')], grid.loc[grid.source.eq('MC')]
        rows.append(dict(candidate=key, nominal_approved=bool(fixed.loc['DATA', 'approved']),
                         DATA_kappa=fixed.loc['DATA', 'kappa'], DATA_kappa_error=fixed.loc['DATA', 'kappa_error'],
                         MC_kappa=fixed.loc['MC', 'kappa'], MC_kappa_error=fixed.loc['MC', 'kappa_error'],
                         nominal_max_S_B=fixed.loc['MC', 'max_S_B'], scan_points=len(data),
                         data_approved_points=int(data.approved.sum()), scan_kappa_min=data.kappa.min(),
                         scan_kappa_max=data.kappa.max(), scan_kappa_span=data.kappa.max()-data.kappa.min(),
                         scan_max_S_B=mc.max_S_B.max(), scan_min_MC_neff=mc.neff_min.min(),
                         scan_min_DATA_count=data[list('ABCD')].min().min(),
                         scan_closure_diagnostic_pass_points=int(data.point_pass.sum())))
    return pd.DataFrame(rows)


def run_candidate_comparison(config):
    specs, definitions = candidate_specs(config)
    planes, signals, manifest, sr = load_candidate_mc(specs, config)
    gates, signal_table, purity_table = candidate_purity(planes, signals, specs, config)
    distribution_purity, distribution_table = candidate_distribution_purity(planes, signals, specs, config)
    data_planes, data_manifest = load_candidate_data(specs, config, gates)
    tables = collect_candidate_tables(planes, data_planes, gates, specs, config)
    tables['distribution_bin_purity'] = distribution_table
    tables['validation'] = validate_candidate_tables(tables, specs, config)
    tables['summary'] = candidate_summary(tables, specs)
    tables['signal_points'] = signal_table
    tables['purity_grid_regions'] = purity_table
    tables['purity_regions'] = pd.concat([purity_table.loc[purity_table.plane.eq(key)
        & np.isclose(purity_table.tx, spec['tx']) & np.isclose(purity_table.ty, spec['ty'])]
        for key, spec in specs.items()], ignore_index=True)
    tables['selection'] = pd.DataFrame([dict(plane=key, **spec) for key, spec in specs.items()])
    sr_reference = collect_mc_sr_reference(sr, config)
    tables.update({'sr_reference_' + name: frame for name, frame in sr_reference['tables'].items()})
    source_manifest = [candidate_input_record(config['sr_reference_dir'] / (name + '.csv'), 'Saved MC SR reference')
                       for name in ('closure', 'scan', 'grid')]
    source_manifest.append(candidate_input_record(mc_sr_full_grid_path(config), '15/16 full MC SR grid reference'))
    for source_path in (Path(__file__), Path(__file__).with_name('abcd_finalize_plots.py')):
        source_manifest.append(candidate_input_record(source_path, 'Analysis implementation'))
    tables['input_manifest'] = pd.DataFrame(manifest + data_manifest + source_manifest)
    selected = tables['selection'][['plane', 'tx', 'ty']].copy()
    displayed_data = {key: plane if gates[key, specs[key]['tx'], specs[key]['ty']]['approved'] else None
                      for key, plane in data_planes.items()}
    return dict(specs=specs, definitions=definitions, selected=selected, planes=planes,
                data_planes=displayed_data, distribution_purity=distribution_purity,
                tables=tables, sr_reference=sr_reference)


def export_candidate_comparison(result, config, figures):
    output = config['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for order, (figure, stem) in enumerate(figures, 1):
        figure.savefig(output / (stem + '.png'), dpi=160, bbox_inches='tight')
        records.append(dict(figure=stem, png=stem + '.png', order=order))
    result['tables']['figure_manifest'] = pd.DataFrame(records)
    for name, frame in result['tables'].items():
        frame.to_csv(output / (name + '.csv'), index=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), settings=config,
                    definitions=result['definitions'], candidate_specs=result['specs'],
                    mc_sr_reference=result['sr_reference']['specs'],
                    diagnostic_scan='Fixed WP (0.30, 2.6); 11 native iso thresholds in (0,2] x 14 dphi thresholds in (0,pi], 154 grid points. Axes start at zero; degenerate zero boundaries are excluded. No DATA-based WP selection.',
                    data_access='VR1 [0,150) DD/PP/DP/PD; VR2 [150,300) PP/DP/PD. No nominal DD SR DATA projections. Failed DATA scan points, nominal distributions and correlations are masked.',
                    purity_policy='120 distinct 200–1000 GeV signal hypotheses, both generated decay topologies; each ABCD S/B_MC strictly <0.10. No leakage veto. No sum of signal hypotheses.',
                    distribution_blinding='Additional DATA VR display mask: each exact displayed bin, after the same isolation-tail and angle-flow folding, requires positive MC background >1e-8 and all 120 finite nonnegative signal yields with max(S/B_MC)<0.10. Failed bins are hidden before color normalization. distribution_bin_purity.csv contains MC only. Integrated ABCD gates, yields and closure are unchanged; this is distribution-display blinding, not removal of information from the existing integrated scan tables.',
                    normalization='Stored production weights; S/B uses MC central values with MC statistical errors, not confidence bounds.',
                    flow_policy='Isolation and mass underflows excluded. Displacement and angle flows retained. Unbounded upper selections retain overflow.',
                    uncertainty='Independent ABCD sumw2 delta method; low statistics and zero-sideband cases flagged.',
                    interpretation='Disjoint displacement categories and mass windows. Nearby boundary scans reuse events. Closure is a diagnostic, never a selection criterion.',
                    validation='Whitelist, SR disjointness, exact native boundaries, projection sumw/sumw2 conservation, DATA masking, and MC SR regression against 11/12 plus the 15/16 grid for iso<=1. Independent native-bin partition validation covers every MC SR point, including the new iso=2 boundary.',
                    figures=records)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + '\n')
    print('Saved:', output)


def mc_sr_spec(config):
    """The common nominal SR from 11/12, independent of candidate VR cuts."""
    return dict(state0='D', state1='D', partner_iso_low=0., partner_iso_high=.5,
                mass_low=150., mass_high=None, tx=.30, ty=2.6,
                scan_mu_iso=list(config['scan_mu_iso']), scan_abs_dphi=list(config['scan_abs_dphi']),
                plateau_mu_iso=[], plateau_abs_dphi=[], kind='mu_dphi', label='Nominal MC SR: D / D',
                selection='0 ≤ partner iso < 0.5; M ≥ 150 GeV',
                x_label='Mu-LJ isolation' if config['topology'] == '2mu2e' else 'Leading mu-LJ isolation',
                y_label='|Δφ| [rad]')


def project_mc_sr(weights, variances, edges, config):
    """MC-only nominal DD projection; the DATA candidate projector stays disjoint."""
    partner = slice(_comparison_split(edges['iso1'], 0.), _comparison_split(edges['iso1'], .5))
    first = slice(None, _comparison_split(edges['disp0'], 2.5))
    second = (slice(_comparison_split(edges['disp1'], .5), None) if config['topology'] == '2mu2e'
              else slice(None, _comparison_split(edges['disp1'], 2.5)))
    mass = slice(_comparison_split(edges['ljlj_mass'], 150.), None)
    selected = [array[:, partner, first, second, :, mass] for array in (weights, variances)]
    _comparison_validate_arrays(*selected)
    reduced = [array.sum(axis=(1, 2, 3, 5)) for array in selected]
    for original, plane in zip(selected, reduced):
        np.testing.assert_allclose(plane.sum(), original.sum(), rtol=1e-10, atol=1e-8)
    return dict(_comparison_plane(*reduced, edges['iso0'], edges['abs_dphi']), kind='mu_dphi')


def collect_mc_sr_reference(sr, config):
    spec, plane = sr['spec'], sr['plane']
    specs, planes = {'SR': spec}, {'SR': {'SR': plane}}
    gates, signals, purity = candidate_purity(planes, {'SR': sr['signals']}, specs, config,
                                             context='SR', require_nominal=False)
    scans, points = candidate_coordinates(spec, plane)
    rows = []
    for point in points:
        tx, ty = point['tx'], point['ty']
        stats = plane_statistics(plane, tx, ty, min_neff=config['min_neff'])
        rows.append(dict(plane='SR', source='MC', context='SR', **stats, **gates['SR', tx, ty],
                         in_plateau=False, point_pass=False))
    grid = pd.DataFrame(rows)
    indexed = grid.set_index(['tx', 'ty'], drop=False)
    scan = pd.DataFrame([dict(indexed.loc[(point['tx'], point['ty'])],
                              scan_axis=point['scan_axis'], threshold=point['threshold']) for point in scans])
    closure = grid.loc[np.isclose(grid.tx, spec['tx']) & np.isclose(grid.ty, spec['ty'])].copy()
    tables = dict(closure=closure, scan=scan, grid=grid, signal_points=signals,
                  purity_grid_regions=purity,
                  purity_regions=purity.loc[np.isclose(purity.tx, spec['tx']) & np.isclose(purity.ty, spec['ty'])].copy(),
                  selection=pd.DataFrame([dict(plane='SR', **spec)]))
    tables['regression'] = validate_mc_sr_reference(tables, config)
    tables['partition_validation'] = validate_mc_sr_partition(plane, grid, tables['regression'])
    return dict(specs=specs, planes=planes, selected=tables['selection'][['plane', 'tx', 'ty']], tables=tables)


def validate_mc_sr_reference(tables, config):
    """Compare cached iso<=1 points; iso=2 has separate native-bin validation."""
    records = []
    metrics = ([field + region for region in 'ABCD' for field in ('', 'var_', 'neff_')]
               + ['pred', 'pred_error', 'kappa', 'kappa_error', 'neff_min'])
    for table in ('closure', 'scan', 'grid'):
        saved = pd.read_csv(config['sr_reference_dir'] / (table + '.csv'))
        saved = saved.loc[saved.plane.eq('mu_dphi') & saved.source.eq('MC') & saved.context.eq('SR')]
        actual = tables[table]
        if saved.empty:
            raise ValueError(f'Missing saved MC SR {table} reference.')
        for row in actual.to_dict('records'):
            rows = saved.loc[np.isclose(saved.tx, row['tx']) & np.isclose(saved.ty, row['ty'])]
            if table == 'scan':
                rows = rows.loc[rows.scan_axis.eq(row['scan_axis'])]
            reference_path = config['sr_reference_dir'] / (table + '.csv')
            if rows.empty and table == 'grid':
                reference_path = mc_sr_full_grid_path(config)
                full = pd.read_csv(reference_path)
                rows = full.loc[full.plane.eq('SR') & full.source.eq('MC') & full.context.eq('SR')
                                & np.isclose(full.tx, row['tx']) & np.isclose(full.ty, row['ty'])]
            if rows.empty and row['tx'] == 2. and table in ('scan', 'grid'):
                # Previous tables stop at iso=1; the new boundary is covered by partition_validation.
                continue
            if len(rows) != 1:
                raise ValueError(f'MC SR reference coordinate mismatch: {table}, {row["tx"]}, {row["ty"]}')
            reference = rows.iloc[0]
            for metric in metrics:
                np.testing.assert_allclose(row[metric], reference[metric], rtol=1e-8, atol=1e-8, equal_nan=True,
                                           err_msg=f'MC SR {table}/{metric}')
                records.append(dict(table=table, tx=reference['tx'], ty=reference['ty'], metric=metric,
                                    recalculated=row[metric], cached=reference[metric],
                                    reference_path=str(reference_path), passed=True))
    return pd.DataFrame(records)


def validate_finalize_config(config):
    """Reject inclusive displacement or nominal-overlapping VR definitions early."""
    if config['topology'] not in ('2mu2e', '4mu'):
        raise ValueError('Unsupported analysis channel.')
    definitions = {'VR1': (0., 150., {'DD', 'PP', 'DP', 'PD'}),
                   'VR2': (150., 300., {'PP', 'DP', 'PD'})}
    if config['vr'] not in definitions:
        raise ValueError('Only VR1 or VR2 may be specified.')
    low, high, states = definitions[config['vr']]
    order = config['candidate_order']
    if len(order) != len(states) or set(order) != states:
        raise ValueError('VR1 allows only DD/PP/DP/PD; VR2 allows only PP/DP/PD.')
    if (config['mass_min'], config['mass_max']) != (low, high):
        raise ValueError('Mass boundaries must remain VR1 [0,150) and VR2 [150,300).')
    if (config.get('partner_iso_low', 0.), config.get('partner_iso_high', .5)) != (0., .5):
        raise ValueError('The common partner isolation interval is [0, 0.5).')
    if (config['tx'], config['ty']) != (.3, 2.6):
        raise ValueError('The common fixed WP is (0.3, 2.6).')
    for field, allowed, nominal in [('scan_mu_iso', {.05, .1, .15, .2, .25, .3, .4, .5, .75, 1., 2.}, .3),
                                    ('scan_abs_dphi', {.5, 1., 1.5, 2., 2.2, 2.4, 2.6, 2.7, 2.8,
                                                      2.9, 3., 3.05, 3.1, np.pi}, 2.6)]:
        values = config[field]
        if len(values) != len(set(values)) or not set(values) <= allowed or nominal not in values:
            raise ValueError(f'{field}: native scan coordinates must include the fixed WP.')
        if list(values) != sorted(values):
            raise ValueError(f'{field}: scan coordinates must be in ascending order.')
    if config['purity_limit'] != .1:
        raise ValueError('DATA display requires the S/B < 0.10 criterion.')


def validate_vr_spec(spec, config):
    validate_finalize_config(config)
    state = spec['state0'] + spec['state1']
    if state not in config['candidate_order'] or spec['state0'] not in ('P', 'D') or spec['state1'] not in ('P', 'D'):
        raise ValueError('Only explicitly allowed P/D combinations may be projected; VR2 DD is forbidden.')
    if (spec['mass_low'], spec['mass_high']) != (config['mass_min'], config['mass_max']):
        raise ValueError('The VR specification does not match the fixed mass interval.')
    if (spec['partner_iso_low'], spec['partner_iso_high']) != (0., .5):
        raise ValueError('All VRs require 0 <= partner iso < 0.5.')
    if (spec['tx'], spec['ty']) != (config['tx'], config['ty']):
        raise ValueError('All VRs must use the common fixed WP.')


def mc_sr_full_grid_path(config):
    directory = ('15_L1_L3_2mu2e_VR_summary_outputs' if config['topology'] == '2mu2e'
                 else '16_L1_L3_4mu_VR_summary_outputs')
    return config['sr_reference_dir'].parent / directory / 'sr_reference_grid.csv'


def validate_mc_sr_partition(plane, grid, regression):
    """Independently sum native-bin masks, including both angle flow bins."""
    xlow = np.r_[-np.inf, plane['xedges']][:, None]
    ylow = np.r_[-np.inf, plane['yedges']][None, :]
    support = xlow >= 0.
    cached_grid = regression.loc[regression.table.eq('grid')]
    cached_points = set(zip(cached_grid.tx, cached_grid.ty))
    records = []
    for row in grid.to_dict('records'):
        low_iso, high_angle = xlow < row['tx'], ylow >= row['ty']
        masks = dict(A=support & low_iso & high_angle, B=support & ~low_iso & high_angle,
                     C=support & low_iso & ~high_angle, D=support & ~low_iso & ~high_angle)
        for key, prefix in [('values', ''), ('variances', 'var_')]:
            for region, mask in masks.items():
                expected = plane[key][mask].sum()
                np.testing.assert_allclose(row[prefix + region], expected, rtol=1e-8, atol=1e-8,
                                           err_msg=f'MC SR native-bin {key}/{region}/{row["tx"]}/{row["ty"]}')
            np.testing.assert_allclose(sum(row[prefix + region] for region in 'ABCD'),
                                       plane[key][1:, :].sum(), rtol=1e-8, atol=1e-8)
        records.append(dict(tx=row['tx'], ty=row['ty'],
                            cached_reference_available=(row['tx'], row['ty']) in cached_points,
                            check='Native-bin ABCD sumw/sumw2 and full-plane partition', passed=True))
    return pd.DataFrame(records)


def candidate_distribution_purity(planes, signals, specs, config):
    """Build MC-only DATA masks in exactly the displayed, flow-folded bins."""
    # Reuse the actual display transform so tails cannot enter a different bin
    # in the purity calculation and the figure. The plotting module loads no MC.
    backgrounds = [planes[key]['VR'] for key in specs]
    xedges = _opt_distribution_edges(backgrounds, 'xedges', 2.)
    yedges = _opt_distribution_edges(backgrounds, 'yedges', np.pi)
    masks, records = {}, []
    for key in specs:
        background = planes[key]['VR']
        b, _ = _opt_fold_distribution(background, xedges, yedges, True)
        vb, _ = _opt_fold_distribution(dict(background, values=background['variances']), xedges, yedges, True)
        samples = sorted(signals[key])
        if len(samples) != 120:
            raise ValueError(f'{key}: displayed-bin blinding requires all 120 signal hypotheses.')
        yields, variances = [], []
        for sample in samples:
            signal = signals[key][sample]
            yields.append(_opt_fold_distribution(signal, xedges, yedges, True)[0])
            variances.append(_opt_fold_distribution(dict(signal, values=signal['variances']), xedges, yedges, True)[0])
        yields, variances = np.asarray(yields), np.asarray(variances)
        valid_background = np.isfinite(b) & (b > 1e-8)
        valid_signal = np.isfinite(yields) & (yields >= 0.)
        valid = valid_signal & valid_background[None, :, :]
        ratios = np.full(yields.shape, np.nan)
        np.divide(yields, b[None, :, :], out=ratios, where=valid)
        finite = np.isfinite(ratios)
        worst = np.argmax(np.where(finite, ratios, -np.inf), axis=0)
        xx, yy = np.indices(b.shape)
        maximum = ratios[worst, xx, yy]
        approved = valid_background & valid_signal.all(axis=0) & finite.all(axis=0) & (maximum < config['purity_limit'])
        masks[key] = dict(xedges=xedges.copy(), yedges=yedges.copy(), approved=approved.copy(),
                          max_S_B=maximum.copy())
        for ix, iy in np.ndindex(b.shape):
            sample_index = worst[ix, iy]
            ratio = maximum[ix, iy]
            error = np.nan
            if np.isfinite(ratio):
                s, vs = yields[sample_index, ix, iy], variances[sample_index, ix, iy]
                error = np.sqrt(vs / b[ix, iy]**2 + s*s*vb[ix, iy] / b[ix, iy]**4)
            reason = ('INVALID_OR_NONPOSITIVE_BACKGROUND' if not valid_background[ix, iy] else
                      'INVALID_SIGNAL_SUPPORT' if not valid_signal[:, ix, iy].all() else
                      'S/B >= 0.10 OR NONFINITE_RATIO' if not approved[ix, iy] else '')
            records.append(dict(plane=key, context='VR', bin_x=ix, bin_y=iy,
                                x_low=xedges[ix], x_high=xedges[ix + 1],
                                y_low=yedges[iy], y_high=yedges[iy + 1],
                                background=b[ix, iy], background_variance=vb[ix, iy],
                                max_S_B=ratio, max_S_B_error=error,
                                worst_sample=samples[sample_index] if np.isfinite(ratio) else '',
                                signal_hypotheses=len(samples),
                                valid_signal_hypotheses=int(valid_signal[:, ix, iy].sum()),
                                valid_background=bool(valid_background[ix, iy]),
                                approved=bool(approved[ix, iy]), gate_reason=reason))
    return masks, pd.DataFrame(records)


# ========================================================================
# review_mc_sr_working_points: original study API
# ========================================================================
"""Compare MC-only nominal SR significance, then inspect identical VR2 WPs.

No nominal DATA is loaded. Each signal hypothesis is evaluated separately with
the normalization stored in the existing production. The shortlist is selected
using nominal MC only; the VR closure tables are diagnostics after selection.
"""

import argparse as finalize_review_mc_sr_working_points_argparse
import hashlib as finalize_review_mc_sr_working_points_hashlib
import json as finalize_review_mc_sr_working_points_json
from pathlib import Path as finalize_review_mc_sr_working_points_Path

import matplotlib as finalize_review_mc_sr_working_points_matplotlib
finalize_review_mc_sr_working_points_matplotlib.use("Agg")
import matplotlib.pyplot as finalize_review_mc_sr_working_points_plt
from matplotlib.backends.backend_pdf import PdfPages as finalize_review_mc_sr_working_points_PdfPages
from matplotlib.colors import TwoSlopeNorm as finalize_review_mc_sr_working_points_TwoSlopeNorm
from matplotlib.patches import Rectangle as finalize_review_mc_sr_working_points_Rectangle
import numpy as finalize_review_mc_sr_working_points_np
import pandas as finalize_review_mc_sr_working_points_pd


finalize_review_mc_sr_working_points_REFERENCE_WP = (.3, 2.6)
finalize_review_mc_sr_working_points_LOCAL_ISO = (.2, .25, .3, .4, .5)
finalize_review_mc_sr_working_points_LOCAL_DPHI = (2.4, 2.6, 2.7, 2.8)
finalize_review_mc_sr_working_points_INPUTS = {"2mu2e": "03_2mu2e_VR2_summary_outputs",
          "4mu": "04_4mu_VR2_summary_outputs"}
finalize_review_mc_sr_working_points_LABELS = {"2mu2e": "2μ2e", "4mu": "4μ"}
finalize_review_mc_sr_working_points_SOURCE = "https://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf"


def finalize_review_mc_sr_working_points_asimov_significance(signal, background, variance=None):
    """Known-background Z or the effective on/off auxiliary-count diagnostic.

    The small-s/b series integrates log(1+r)-log(1+alpha*r), where alpha is
    v/(b+v). It avoids cancellation without imposing a background floor.
    Negative signal, nonpositive background and invalid variance return NaN.
    """
    s, b = finalize_review_mc_sr_working_points_np.broadcast_arrays(finalize_review_mc_sr_working_points_np.asarray(signal, dtype=finalize_review_mc_sr_working_points_np.longdouble),
                               finalize_review_mc_sr_working_points_np.asarray(background, dtype=finalize_review_mc_sr_working_points_np.longdouble))
    v = finalize_review_mc_sr_working_points_np.zeros_like(b) if variance is None else finalize_review_mc_sr_working_points_np.broadcast_to(
        finalize_review_mc_sr_working_points_np.asarray(variance, dtype=finalize_review_mc_sr_working_points_np.longdouble), b.shape)
    valid = finalize_review_mc_sr_working_points_np.isfinite(s) & finalize_review_mc_sr_working_points_np.isfinite(b) & finalize_review_mc_sr_working_points_np.isfinite(v) & (s >= 0) & (b > 0) & (v >= 0)
    result = finalize_review_mc_sr_working_points_np.full(s.shape, finalize_review_mc_sr_working_points_np.nan, dtype=finalize_review_mc_sr_working_points_np.longdouble)
    if not finalize_review_mc_sr_working_points_np.any(valid):
        return finalize_review_mc_sr_working_points_np.asarray(result, dtype=float)
    r, bv, vv = s[valid] / b[valid], b[valid], v[valid]
    t = vv / bv
    alpha = t / (1 + t)
    small = r < finalize_review_mc_sr_working_points_np.longdouble("0.0001")
    f = finalize_review_mc_sr_working_points_np.zeros_like(r)
    for n in range(1, 9):
        # 1-alpha**n is evaluated as (1-alpha)*sum(alpha**k).
        coefficient = sum(alpha[small] ** k for k in range(n)) / (1 + t[small])
        f[small] += (-1) ** (n + 1) * coefficient * r[small] ** (n + 1) / (n * (n + 1))
    known = (~small) & (vv == 0)
    f[known] = (1 + r[known]) * finalize_review_mc_sr_working_points_np.log1p(r[known]) - r[known]
    uncertain = (~small) & (vv > 0)
    ru, tu, au = r[uncertain], t[uncertain], alpha[uncertain]
    f[uncertain] = (1 + ru) * (finalize_review_mc_sr_working_points_np.log1p(ru) - finalize_review_mc_sr_working_points_np.log1p(au * ru)) - finalize_review_mc_sr_working_points_np.log1p(au * ru) / tu
    # Only round-off sized negative values can be repaired; invalid values stay NaN.
    tolerance = 64 * finalize_review_mc_sr_working_points_np.finfo(finalize_review_mc_sr_working_points_np.longdouble).eps * finalize_review_mc_sr_working_points_np.maximum(1, r * r)
    f[(f < 0) & (f >= -tolerance)] = 0
    with finalize_review_mc_sr_working_points_np.errstate(invalid="ignore"):
        result[valid] = finalize_review_mc_sr_working_points_np.sqrt(2 * bv * f)
    return finalize_review_mc_sr_working_points_np.asarray(result, dtype=float)


def finalize_review_mc_sr_working_points_validate_significance():
    """Check known limits, a closed-form point, scaling and invalid inputs."""
    exact = finalize_review_mc_sr_working_points_np.sqrt(2 * (30 * finalize_review_mc_sr_working_points_np.log(1.5) - 10))
    finalize_review_mc_sr_working_points_np.testing.assert_allclose(finalize_review_mc_sr_working_points_asimov_significance(10, 20), exact, rtol=2e-14)
    finalize_review_mc_sr_working_points_np.testing.assert_allclose(finalize_review_mc_sr_working_points_asimov_significance([0, 10], 20, 0), [0, exact], rtol=2e-14)
    s = finalize_review_mc_sr_working_points_np.array([1e-12, 1e-7, .01, 1., 10., 1000.])
    known = finalize_review_mc_sr_working_points_asimov_significance(s, 20)
    finalize_review_mc_sr_working_points_np.testing.assert_allclose(finalize_review_mc_sr_working_points_asimov_significance(s, 20, 1e-12), known, rtol=1e-11)
    finalize_review_mc_sr_working_points_np.testing.assert_allclose(known[:2], s[:2] / finalize_review_mc_sr_working_points_np.sqrt(20), rtol=1e-8)
    finalize_review_mc_sr_working_points_np.testing.assert_allclose(finalize_review_mc_sr_working_points_asimov_significance(s[:2], 20, 7), s[:2] / finalize_review_mc_sr_working_points_np.sqrt(27), rtol=1e-8)
    finalize_review_mc_sr_working_points_np.testing.assert_allclose(finalize_review_mc_sr_working_points_asimov_significance(s * 9, 20 * 9, 7 * 9),
                               3 * finalize_review_mc_sr_working_points_asimov_significance(s, 20, 7), rtol=2e-12)
    assert finalize_review_mc_sr_working_points_np.all(finalize_review_mc_sr_working_points_asimov_significance(s, 20, 7) <= known)
    assert finalize_review_mc_sr_working_points_np.isnan(finalize_review_mc_sr_working_points_asimov_significance([1, -1, 1], [0, 1, -1])).all()
    assert finalize_review_mc_sr_working_points_np.isnan(finalize_review_mc_sr_working_points_asimov_significance(1, 1, -1))
    return {"known_background_closed_form": True, "zero_signal": True,
            "zero_variance_limit": True, "small_signal_stability": True,
            "simultaneous_exposure_scaling": True, "invalid_inputs_rejected": True}


def finalize_review_mc_sr_working_points_file_hash(path):
    return finalize_review_mc_sr_working_points_hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_review_mc_sr_working_points_finite_ratio(numerator, denominator):
    numerator, denominator = finalize_review_mc_sr_working_points_np.broadcast_arrays(finalize_review_mc_sr_working_points_np.asarray(numerator, dtype=float),
                                                finalize_review_mc_sr_working_points_np.asarray(denominator, dtype=float))
    ratio = finalize_review_mc_sr_working_points_np.full(numerator.shape, finalize_review_mc_sr_working_points_np.nan)
    valid = finalize_review_mc_sr_working_points_np.isfinite(numerator) & finalize_review_mc_sr_working_points_np.isfinite(denominator) & (denominator > 0)
    finalize_review_mc_sr_working_points_np.divide(numerator, denominator, out=ratio, where=valid)
    return ratio


def finalize_review_mc_sr_working_points_signal_significance(channel, background, signal):
    if not background.source.eq("MC").all() or not background.context.eq("SR").all():
        raise ValueError("Expected only nominal MC SR background rows.")
    if not signal.context.eq("SR").all() or signal["sample"].nunique() != 120:
        raise ValueError("Expected 120 separate nominal MC SR signal hypotheses.")
    if len(background) != 154 or background.duplicated(["tx", "ty"]).any():
        raise ValueError("Expected the existing 154-point MC-only grid.")
    columns = ["tx", "ty", "A", "var_A", "A_error", "neff_A"]
    frame = signal.merge(background[columns].rename(columns={
        "A": "b", "var_A": "var_b", "A_error": "sigma_b", "neff_A": "neff_b"}),
                         on=["tx", "ty"], validate="many_to_one")
    if len(frame) != 154 * 120 or frame.duplicated(["sample", "tx", "ty"]).any():
        raise ValueError("Signal hypotheses must cover every WP exactly once.")
    frame.insert(0, "topology", channel)
    frame["signal_decay"] = frame["sample"].str.split("_").str[0]
    matching = "2Mu2E" if channel == "2mu2e" else "4Mu"
    frame["matching_decay"] = frame.signal_decay.eq(matching)
    if frame.loc[frame.matching_decay, "sample"].nunique() != 60:
        raise ValueError("Expected 60 hypotheses for the matching signal decay.")
    frame["s"] = frame.S_A
    frame["var_s"] = frame.var_S_A
    frame["neff_s"] = finalize_review_mc_sr_working_points_finite_ratio(frame.s ** 2, frame.var_s)
    frame["low_mc_background_stat"] = ~finalize_review_mc_sr_working_points_np.isfinite(frame.neff_b) | (frame.neff_b < 10)
    frame["low_mc_signal_stat"] = ~finalize_review_mc_sr_working_points_np.isfinite(frame.neff_s) | (frame.neff_s < 10)
    frame["z_known_b"] = finalize_review_mc_sr_working_points_asimov_significance(frame.s, frame.b)
    frame["z_mc_stat_diagnostic"] = finalize_review_mc_sr_working_points_asimov_significance(frame.s, frame.b, frame.var_b)
    frame["s_sqrt_b"] = finalize_review_mc_sr_working_points_finite_ratio(frame.s, finalize_review_mc_sr_working_points_np.sqrt(frame.b.where(frame.b > 0)))
    frame["s_sqrt_s_plus_b"] = finalize_review_mc_sr_working_points_finite_ratio(frame.s, finalize_review_mc_sr_working_points_np.sqrt((frame.s + frame.b).where(frame.b > 0)))
    metrics = ["z_known_b", "z_mc_stat_diagnostic", "s_sqrt_b", "s_sqrt_s_plus_b"]
    reference = frame.loc[finalize_review_mc_sr_working_points_np.isclose(frame.tx, finalize_review_mc_sr_working_points_REFERENCE_WP[0]) &
                          finalize_review_mc_sr_working_points_np.isclose(frame.ty, finalize_review_mc_sr_working_points_REFERENCE_WP[1]), ["sample"] + metrics]
    reference = reference.rename(columns={name: name + "_reference" for name in metrics})
    frame = frame.merge(reference, on="sample", validate="many_to_one")
    for name in metrics:
        frame[name + "_ratio"] = finalize_review_mc_sr_working_points_finite_ratio(frame[name], frame[name + "_reference"])
    frame["ratio_status"] = finalize_review_mc_sr_working_points_np.where(frame.z_known_b_reference > 0, "VALID", "ZERO_OR_INVALID_REFERENCE")
    frame["new_positive_signal_from_zero_reference"] = frame.z_known_b_reference.eq(0) & frame.z_known_b.gt(0)
    frame["in_local_scan"] = frame.tx.isin(finalize_review_mc_sr_working_points_LOCAL_ISO) & frame.ty.isin(finalize_review_mc_sr_working_points_LOCAL_DPHI)
    frame["reference_wp"] = finalize_review_mc_sr_working_points_np.isclose(frame.tx, finalize_review_mc_sr_working_points_REFERENCE_WP[0]) & finalize_review_mc_sr_working_points_np.isclose(frame.ty, finalize_review_mc_sr_working_points_REFERENCE_WP[1])
    return frame


def finalize_review_mc_sr_working_points_summarize_significance(frame):
    rows = []
    metrics = ["z_known_b", "z_mc_stat_diagnostic", "s_sqrt_b", "s_sqrt_s_plus_b"]
    for channel, channel_frame in frame.groupby("topology", sort=False):
        sets = {"matching_decay": channel_frame.loc[channel_frame.matching_decay],
                "2Mu2E": channel_frame.loc[channel_frame.signal_decay.eq("2Mu2E")],
                "4Mu": channel_frame.loc[channel_frame.signal_decay.eq("4Mu")],
                "all_120": channel_frame}
        for set_name, subset in sets.items():
            for (tx, ty), point in subset.groupby(["tx", "ty"]):
                first = point.iloc[0]
                row = {"topology": channel, "signal_set": set_name, "tx": tx, "ty": ty,
                       "hypotheses": len(point), "b": first.b, "sigma_b": first.sigma_b,
                       "neff_b": first.neff_b, "low_mc_background_stat": bool(first.low_mc_background_stat),
                       "in_local_scan": bool(first.in_local_scan), "reference_wp": bool(first.reference_wp),
                       "signal_neff_min": point.neff_s.min(),
                       "new_positive_signal_from_zero_reference": int(point.new_positive_signal_from_zero_reference.sum()),
                       "low_mc_signal_stat_hypotheses": int(point.low_mc_signal_stat.sum())}
                for metric in metrics:
                    absolute = point[metric].replace([finalize_review_mc_sr_working_points_np.inf, -finalize_review_mc_sr_working_points_np.inf], finalize_review_mc_sr_working_points_np.nan).dropna()
                    ratio = point[metric + "_ratio"].replace([finalize_review_mc_sr_working_points_np.inf, -finalize_review_mc_sr_working_points_np.inf], finalize_review_mc_sr_working_points_np.nan).dropna()
                    row.update({metric + "_median": absolute.median(),
                                metric + "_min": absolute.min(), metric + "_max": absolute.max(),
                                metric + "_ratio_valid": len(ratio), metric + "_ratio_median": ratio.median(),
                                metric + "_ratio_q10": ratio.quantile(.1),
                                metric + "_ratio_q90": ratio.quantile(.9),
                                metric + "_ratio_worst": ratio.min(),
                                metric + "_improved_hypotheses": int((ratio > 1 + 1e-12).sum())})
                rows.append(row)
    return finalize_review_mc_sr_working_points_pd.DataFrame(rows)


def finalize_review_mc_sr_working_points_rank_and_select(summary):
    primary = summary.loc[summary.signal_set.eq("matching_decay") & summary.in_local_scan].copy()
    primary = primary.sort_values(["topology", "z_known_b_ratio_median", "tx", "ty"],
                                  ascending=[True, False, True, True])
    primary["mc_rank_including_reference"] = primary.groupby("topology").cumcount() + 1
    shortlist = []
    for channel, group in primary.groupby("topology", sort=False):
        if len(group) != 20 or not group.z_known_b_ratio_valid.eq(60).all():
            raise ValueError("Each local WP must have 60 valid matching-decay reference ratios.")
        baseline = group.loc[group.reference_wp].copy()
        baseline["selection"] = "reference"
        baseline["selection_reason"] = "reference"
        baseline["candidate_rank"] = 0
        alternatives = group.loc[~group.reference_wp]
        selected = alternatives.head(2).copy()
        selected["selection_reason"] = "top_two_known_background"
        robust = alternatives.sort_values(["z_mc_stat_diagnostic_ratio_median", "tx", "ty"],
                                          ascending=[False, True, True]).iloc[0]
        if robust.name not in selected.index:
            third = alternatives.loc[[robust.name]].copy()
            third["selection_reason"] = "best_mc_stat_diagnostic"
        else:
            third = alternatives.loc[~alternatives.index.isin(selected.index)].head(1).copy()
            third["selection_reason"] = "next_known_background"
        selected = finalize_review_mc_sr_working_points_pd.concat([selected, third])
        selected["selection"] = "MC_only_candidate"
        selected["candidate_rank"] = finalize_review_mc_sr_working_points_np.arange(1, len(selected) + 1)
        shortlist.extend([baseline, selected])
    return primary, finalize_review_mc_sr_working_points_pd.concat(shortlist, ignore_index=True)


def finalize_review_mc_sr_working_points_shortlisted_vr(shortlist, vr_frames):
    rows = []
    sensitive = ["A", "B", "C", "D", "var_A", "var_B", "var_C", "var_D", "A_error",
                 "B_error", "C_error", "D_error", "neff_A", "neff_B", "neff_C", "neff_D",
                 "neff_min", "min_neff", "min_sideband_neff", "pred", "pred_error", "kappa", "kappa_error", "pull"]
    for channel, vr in vr_frames.items():
        if set(vr.plane) != {"PP", "DP", "PD"} or not vr.context.eq("VR").all():
            raise ValueError("VR2 must contain exactly PP, DP and PD.")
        chosen = shortlist.loc[shortlist.topology.eq(channel),
                               ["topology", "tx", "ty", "candidate_rank", "selection"]]
        merged = chosen.merge(vr, on=["tx", "ty"], validate="one_to_many")
        if len(merged) != 24:
            raise ValueError("Expected MC and DATA for three VR2 states at four selected WPs.")
        # Defense in depth: never export DATA values when the integrated gate failed.
        forbidden = merged.source.eq("DATA") & ~merged.approved.eq(True)
        merged.loc[forbidden, [c for c in sensitive if c in merged]] = finalize_review_mc_sr_working_points_np.nan
        merged["integrated_wp_data_gate"] = merged.approved.eq(True)
        rows.append(merged)
    return finalize_review_mc_sr_working_points_pd.concat(rows, ignore_index=True)


def finalize_review_mc_sr_working_points_plot_gain_maps(ranking, shortlist):
    fig, axes = finalize_review_mc_sr_working_points_plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, channel in zip(axes, finalize_review_mc_sr_working_points_INPUTS):
        frame = ranking.loc[ranking.topology.eq(channel)]
        matrix = frame.pivot(index="ty", columns="tx", values="z_known_b_ratio_median").reindex(
            index=finalize_review_mc_sr_working_points_LOCAL_DPHI, columns=finalize_review_mc_sr_working_points_LOCAL_ISO).to_numpy()
        norm = finalize_review_mc_sr_working_points_TwoSlopeNorm(vmin=min(.99, float(finalize_review_mc_sr_working_points_np.nanmin(matrix))), vcenter=1,
                            vmax=max(1.01, float(finalize_review_mc_sr_working_points_np.nanmax(matrix))))
        mesh = ax.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", norm=norm)
        for iy, ty in enumerate(finalize_review_mc_sr_working_points_LOCAL_DPHI):
            for ix, tx in enumerate(finalize_review_mc_sr_working_points_LOCAL_ISO):
                ax.text(ix, iy, f"{matrix[iy, ix]:.3f}", ha="center", va="center", fontsize=10)
        for point in shortlist.loc[shortlist.topology.eq(channel)].itertuples():
            ix, iy = finalize_review_mc_sr_working_points_LOCAL_ISO.index(point.tx), finalize_review_mc_sr_working_points_LOCAL_DPHI.index(point.ty)
            ax.add_patch(finalize_review_mc_sr_working_points_Rectangle((ix - .47, iy - .47), .94, .94, fill=False,
                                   edgecolor="black" if point.candidate_rank == 0 else "#009E73", lw=2.3))
            if point.candidate_rank:
                ax.text(ix + .38, iy + .35, f"#{point.candidate_rank}", fontsize=8,
                        color="#006644", ha="right", va="top")
        ax.set_xticks(range(len(finalize_review_mc_sr_working_points_LOCAL_ISO)), [f"{v:g}" for v in finalize_review_mc_sr_working_points_LOCAL_ISO])
        ax.set_yticks(range(len(finalize_review_mc_sr_working_points_LOCAL_DPHI)), [f"{v:g}" for v in finalize_review_mc_sr_working_points_LOCAL_DPHI])
        ax.set(xlabel="Muon isolation boundary", ylabel="|Δφ| boundary", title=f"{finalize_review_mc_sr_working_points_LABELS[channel]} · nominal MC SR")
        fig.colorbar(mesh, ax=ax, fraction=.045, label="Median Z / Z(reference), 60 matching hypotheses")
    fig.suptitle("MC SR boundary candidates · expected counting significance", fontsize=13)
    fig.text(.5, .018, "Reference: (iso, |Δφ|) = (0.3, 2.6), black outline. Green: 2 leading WPs plus 1 MC-stat robustness candidate.\n"
             "Each signal hypothesis is evaluated separately. Nominal DATA is never loaded. MC-only ranking; no VR closure selection.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.94, top=.85, bottom=.22, wspace=.38)
    return fig


def finalize_review_mc_sr_working_points_plot_signal_gains(signal, shortlist):
    fig, axes = finalize_review_mc_sr_working_points_plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, channel in zip(axes, finalize_review_mc_sr_working_points_INPUTS):
        selected = shortlist.loc[shortlist.topology.eq(channel)].sort_values("candidate_rank")
        known, diagnostic, labels = [], [], []
        for point in selected.itertuples():
            values = signal.loc[signal.topology.eq(channel) & signal.matching_decay &
                                finalize_review_mc_sr_working_points_np.isclose(signal.tx, point.tx) & finalize_review_mc_sr_working_points_np.isclose(signal.ty, point.ty)]
            known.append(values.z_known_b_ratio.to_numpy())
            diagnostic.append(values.z_mc_stat_diagnostic_ratio.to_numpy())
            labels.append(f"({point.tx:g}, {point.ty:g})\nb={point.b:.2f} ± {point.sigma_b:.2f}\nNeff={point.neff_b:.2f}")
        x = finalize_review_mc_sr_working_points_np.arange(4)
        for data, offset, color, label in [(known, -.13, "#0072B2", "Known background Z"),
                                           (diagnostic, .13, "#E69F00", "MC-stat on/off diagnostic")]:
            quantiles = finalize_review_mc_sr_working_points_np.array([finalize_review_mc_sr_working_points_np.quantile(values[finalize_review_mc_sr_working_points_np.isfinite(values)], [.1, .5, .9]) for values in data])
            ax.errorbar(x + offset, quantiles[:, 1],
                        yerr=finalize_review_mc_sr_working_points_np.vstack([quantiles[:, 1] - quantiles[:, 0], quantiles[:, 2] - quantiles[:, 1]]),
                        fmt="o", color=color, capsize=4, label=label)
        ax.axhline(1, color="black", lw=1, ls="--")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set(title=f"{finalize_review_mc_sr_working_points_LABELS[channel]} · per-hypothesis improvement", ylabel="Z / Z(reference)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Nominal MC SR · same stored signal normalization at every WP", fontsize=13)
    fig.text(.5, .015, "Points: median over 60 matching signal hypotheses; bars: 10th–90th percentile across hypotheses, NOT statistical errors.\n"
             "Background MC effective counts are small. The auxiliary-count diagnostic uses σ²b = sumw²; it is not an ABCD likelihood.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.97, top=.86, bottom=.25, wspace=.25)
    return fig


def finalize_review_mc_sr_working_points_plot_vr_closure(vr, shortlist):
    fig, axes = finalize_review_mc_sr_working_points_plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for row, channel in enumerate(finalize_review_mc_sr_working_points_INPUTS):
        selected = shortlist.loc[shortlist.topology.eq(channel)].sort_values("candidate_rank")
        labels = [f"{'Ref' if p.candidate_rank == 0 else '#' + str(p.candidate_rank)}\n({p.tx:g}, {p.ty:g})"
                  for p in selected.itertuples()]
        for column, state in enumerate(("PP", "DP", "PD")):
            ax = axes[row, column]
            ax.axhspan(.9, 1.1, color="#009E73", alpha=.15)
            ax.axhline(1, color="#009E73", lw=1)
            clipped = 0
            for source, shift, color in [("MC", -.10, "#E69F00"), ("DATA", .10, "#202020")]:
                group = vr.loc[vr.topology.eq(channel) & vr.plane.eq(state) & vr.source.eq(source)]
                group = group.sort_values("candidate_rank")
                for position, point in enumerate(group.itertuples()):
                    if source == "DATA" and not point.integrated_wp_data_gate:
                        ax.text(position + shift, .06, "blind", rotation=90, fontsize=7, ha="center")
                        continue
                    value, error = point.kappa, point.kappa_error
                    if not finalize_review_mc_sr_working_points_np.isfinite(value):
                        ax.text(position + shift, .06, "n/a", rotation=90, fontsize=7, ha="center")
                        continue
                    ax.errorbar(position + shift, value, yerr=error if finalize_review_mc_sr_working_points_np.isfinite(error) else None,
                                color=color, fmt="o", capsize=3, label=source if position == 0 else None)
                    if value > 2 or value < 0:
                        ax.plot(position + shift, 1.97 if value > 2 else .03,
                                "^" if value > 2 else "v", color=color, clip_on=False)
                    if finalize_review_mc_sr_working_points_np.isfinite(error) and (value - error < 0 or value + error > 2):
                        clipped += 1
            ax.set(ylim=(0, 2), xlim=(-.5, 3.5), title=f"{finalize_review_mc_sr_working_points_LABELS[channel]} VR2 {state}")
            ax.set_xticks(range(4), labels, fontsize=8)
            ax.grid(alpha=.25)
            if column == 0:
                ax.set_ylabel("κ = A / (BC/D)")
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
            if clipped:
                ax.text(.02, .96, f"{clipped} error ranges clipped", fontsize=7, transform=ax.transAxes, va="top")
    fig.suptitle("Identical boundaries applied to VR2 · 150 ≤ MJJ < 300 GeV · partner iso < 0.5", fontsize=13)
    fig.text(.5, .012, "Selection uses MC SR significance only. VR DATA is displayed only when the existing all-hypothesis, all-ABCD S/B < 10% WP gate passes.\n"
             "Errors are first-order propagation of independent A/B/C/D sumw². Large sparse-count uncertainties limit closure conclusions.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.06, right=.98, top=.90, bottom=.15, hspace=.42, wspace=.18)
    return fig


def finalize_review_mc_sr_working_points_run_review(base, output):
    checks = finalize_review_mc_sr_working_points_validate_significance()
    input_hashes, signals, vr_frames = {}, [], {}
    for channel, directory in finalize_review_mc_sr_working_points_INPUTS.items():
        paths = {name: base / directory / (name + ".csv") for name in
                 ("sr_reference_grid", "sr_reference_signal_points", "grid")}
        frames = {name: finalize_review_mc_sr_working_points_pd.read_csv(path) for name, path in paths.items()}
        input_hashes.update({str(path.relative_to(base)): finalize_review_mc_sr_working_points_file_hash(path) for path in paths.values()})
        signals.append(finalize_review_mc_sr_working_points_signal_significance(channel, frames["sr_reference_grid"], frames["sr_reference_signal_points"]))
        vr_frames[channel] = frames["grid"]
    signal = finalize_review_mc_sr_working_points_pd.concat(signals, ignore_index=True)
    summary = finalize_review_mc_sr_working_points_summarize_significance(signal)
    ranking, shortlist = finalize_review_mc_sr_working_points_rank_and_select(summary)
    vr = finalize_review_mc_sr_working_points_shortlisted_vr(shortlist, vr_frames)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"mc_sr_signal_significance": signal, "mc_sr_working_point_summary": summary,
              "mc_sr_local_ranking": ranking, "mc_sr_shortlist": shortlist, "vr2_shortlist_closure": vr}
    for name, frame in tables.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    figures = [("mc_sr_local_significance_gain", finalize_review_mc_sr_working_points_plot_gain_maps(ranking, shortlist)),
               ("mc_sr_shortlist_signal_gains", finalize_review_mc_sr_working_points_plot_signal_gains(signal, shortlist)),
               ("vr2_shortlist_closure", finalize_review_mc_sr_working_points_plot_vr_closure(vr, shortlist))]
    for name, fig in figures:
        fig.savefig(output / (name + ".png"), dpi=160)
        finalize_review_mc_sr_working_points_plt.close(fig)
    metadata = {
        "reference_wp": {"iso": finalize_review_mc_sr_working_points_REFERENCE_WP[0], "abs_dphi": finalize_review_mc_sr_working_points_REFERENCE_WP[1]},
        "nominal_selection": "DD, MJJ >= 150 GeV, 0 <= partner iso < 0.5; A: iso < tx and abs_dphi >= ty",
        "vr_selection": "VR2 only: 150 <= MJJ < 300 GeV, PP/DP/PD, 0 <= partner iso < 0.5",
        "nominal_data_loaded": False, "nominal_data_A_evaluated": False,
        "normalization": "Stored production MC weights, unchanged; no common cross-section rescaling; existing luminosity and production cross sections retained.",
        "full_grid_points_per_channel": 154,
        "primary_local_scan": {"iso": list(finalize_review_mc_sr_working_points_LOCAL_ISO), "abs_dphi": list(finalize_review_mc_sr_working_points_LOCAL_DPHI), "points": 20},
        "primary_metric": "Z = sqrt(2*((s+b)*log1p(s/b)-s)), with s=nominal signal MC A and b=nominal background MC A",
        "primary_ranking": "Descending median of per-hypothesis Z(WP)/Z(reference), 60 matching-decay hypotheses per channel.",
        "shortlist_policy": "Top two non-reference known-background WPs, plus the best MC-stat on/off diagnostic WP if distinct; otherwise the next known-background WP. Selection uses MC only.",
        "zero_reference_policy": "Z(reference)=0 yields an undefined ratio, never infinity; newly positive hypotheses are counted separately. Every ratio summary uses the same nonzero-reference cohort within that signal set.",
        "ranking_uses_vr_closure_or_data": False,
        "signal_sets": "Matching decay is primary; separate 2Mu2E, 4Mu and all-120 summaries are diagnostics. Signal hypotheses are never summed.",
        "diagnostic_metric": "Effective on/off Asimov significance, variance v=sumw2(background MC A), Cowan Eq.20; v=0 gives the known-background limit.",
        "statistical_source": {"url": finalize_review_mc_sr_working_points_SOURCE, "equations": [10, 20]},
        "validation": checks,
        "limitations": [
            "Exploratory expected counting-significance proxy, not a full ABCD discovery likelihood or a final WP choice.",
            "Known-background ranking excludes background uncertainty; the effective on/off result is a diagnostic approximation for weighted MC.",
            "No background closure nuisance, signal MC uncertainty, or signal leakage into B/C/D is modeled in the significance.",
            "All local candidate background MC effective counts are below 10; asymptotic numerical values require caution.",
            "2mu2e iso=0.2 improvement is driven by losing the large QCD_Pt170To300 contribution present at the reference (sumw=21.982536, Neff=3 in an independent raw-MC audit).",
            "4mu candidate background is dominated by one effective DYJetsToLL_M50 event (sumw=3.2480395 in an independent raw-MC audit).",
            "VR2 DATA retains the integrated all-hypothesis/all-ABCD S/B<0.1 gate. Event-bin distribution blinding is implemented separately and does not alter these integrated closure counts.",
            "The scan has no trials correction. Closely spaced WPs share events and are correlated.",
            "Background<=0 gives undefined significance; no artificial background floor is used."
        ],
        "inputs_sha256": input_hashes,
        "script_sha256": finalize_review_mc_sr_working_points_file_hash(finalize_review_mc_sr_working_points_Path(__file__)),
        "tables_rows": {name: len(frame) for name, frame in tables.items()},
    }
    (output / "methodology.json").write_text(finalize_review_mc_sr_working_points_json.dumps(metadata, indent=2) + "\n")
    columns = ["topology", "candidate_rank", "tx", "ty", "b", "sigma_b", "neff_b",
               "z_known_b_ratio_median", "z_known_b_ratio_q10", "z_known_b_ratio_q90",
               "z_mc_stat_diagnostic_ratio_median"]
    print(shortlist[columns].to_string(index=False))
    print(f"Saved {len(tables)} tables and {len(figures)} figures to {output}")


def finalize_review_mc_sr_working_points_main():
    parser = finalize_review_mc_sr_working_points_argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=finalize_review_mc_sr_working_points_Path, default=finalize_review_mc_sr_working_points_Path(__file__).resolve().parent)
    parser.add_argument("--output", type=finalize_review_mc_sr_working_points_Path)
    args = parser.parse_args()
    finalize_review_mc_sr_working_points_run_review(args.base, args.output or args.base / "MC_SR_WP_review_outputs")


# ========================================================================
# abcd_local_scan: original study API
# ========================================================================
"""MC-ranked local WP summaries with cached, blinded VR validation tables.

Only the 18 original background-MC files are opened. Signal yields and gated
VR DATA statistics come from the validated 01/03 exports. Nominal DATA and
original signal/DATA coffea files are never loaded by this module.
"""
from datetime import datetime as finalize_local_scan_datetime; from datetime import timezone as finalize_local_scan_timezone
import gc as finalize_local_scan_gc
import json as finalize_local_scan_json
from pathlib import Path as finalize_local_scan_Path

import numpy as finalize_local_scan_np
import pandas as finalize_local_scan_pd

finalize_local_scan_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_local_scan_review = _FINALIZE_NAMESPACES['review_mc_sr_working_points']


finalize_local_scan_LOCAL_ISO = (.20, .25, .30, .40)
finalize_local_scan_LOCAL_DPHI = (2.4, 2.6, 2.7, 2.8)
finalize_local_scan_REFERENCE_WP = (.30, 2.6)
finalize_local_scan_SCOPE_ORDER = ('SR', 'VR1_DD', 'VR1_PP', 'VR1_DP', 'VR1_PD',
               'VR2_PP', 'VR2_DP', 'VR2_PD')
finalize_local_scan_REGIONS = tuple('ABCD')
finalize_local_scan_STATISTICS = (list('ABCD') + ['var_' + region for region in finalize_local_scan_REGIONS]
              + ['pred', 'pred_error', 'kappa', 'kappa_error'])


def finalize_local_scan_build_config(notebook_dir, output_dir=None):
    """Use the unchanged 2mu2e production, selection and histogram conventions."""
    directory = finalize_local_scan_Path(notebook_dir).resolve()
    inputs = directory.parent / 'ABCD_plane_summary'
    return dict(
        notebook_dir=directory, topology='2mu2e',
        output_dir=finalize_local_scan_Path(output_dir) if output_dir is not None else directory / '07_2mu2e_LocalScan_summary_outputs',
        notebook_path=directory / '07_MuIso_DPhi_2mu2e_LocalScan_summary.ipynb',
        bkg_dir=inputs / 'ABCD_6D_inclusive_eHEM_skimv1_v1_bkg_merged_samples_v1',
        current_hist='abcd6d_2mu2e',
        expected_current_channel='abcd6d_inclusive_2mu2e_dxySpread_cosAlpha_mu_veto_eHEM',
        vr1_output_dir=directory / '01_2mu2e_VR1_summary_outputs',
        vr2_output_dir=directory / '03_2mu2e_VR2_summary_outputs',
        sr_reference_dir=inputs / '11_Optimized_ABCD_2mu2e_comparison_outputs',
        local_iso=list(finalize_local_scan_LOCAL_ISO), local_dphi=list(finalize_local_scan_LOCAL_DPHI), top_n=5,
        reference_wp=finalize_local_scan_REFERENCE_WP, partner_iso_low=0., partner_iso_high=.5,
        tx=.30, ty=2.6, min_neff=10., purity_limit=.10, closure_band=(.9, 1.1),
        scan_mu_iso=[.05, .1, .15, .2, .25, .3, .4, .5, .75, 1., 2.],
        scan_abs_dphi=[.5, 1., 1.5, 2., 2.2, 2.4, 2.6, 2.7, 2.8,
                       2.9, 3., 3.05, 3.1, finalize_local_scan_np.pi],
        save_all_png=False)


def finalize_local_scan__vr_config(config, vr):
    states = ['DD', 'PP', 'DP', 'PD'] if vr == 'VR1' else ['PP', 'DP', 'PD']
    low, high = (0., 150.) if vr == 'VR1' else (150., 300.)
    result = dict(config, vr=vr, candidate_order=states, mass_min=low, mass_max=high)
    finalize_local_scan_common.validate_finalize_config(result)
    return result


def finalize_local_scan__check_local_config(config):
    if config['topology'] != '2mu2e':
        raise ValueError('Notebook 07 is restricted to the 2mu2e channel.')
    if tuple(config['local_iso']) != finalize_local_scan_LOCAL_ISO or tuple(config['local_dphi']) != finalize_local_scan_LOCAL_DPHI:
        raise ValueError('Use the 16 native WPs in iso=[0.20,0.40] and dphi=[2.4,2.8].')
    if tuple(config['reference_wp']) != finalize_local_scan_REFERENCE_WP or config['top_n'] != 5:
        raise ValueError('Keep reference=(0.30,2.6) and the MC-ranked top-five display.')
    if (config['partner_iso_low'], config['partner_iso_high']) != (0., .5):
        raise ValueError('The common partner-isolation interval must remain [0,0.5).')
    for vr in ('VR1', 'VR2'):
        finalize_local_scan__vr_config(config, vr)


def finalize_local_scan__coordinates(config):
    records = []
    for tx in config['local_iso']:
        for ty in config['local_dphi']:
            records.append(dict(wp=f'W{len(records) + 1:02d}', tx=float(tx), ty=float(ty),
                                label=f'iso={tx:.2f}, Δφ={ty:.1f}',
                                is_reference=bool(finalize_local_scan_np.isclose(tx, .30) and finalize_local_scan_np.isclose(ty, 2.6))))
    frame = finalize_local_scan_pd.DataFrame(records)
    if len(frame) != 16 or frame.duplicated(['tx', 'ty']).any() or frame.is_reference.sum() != 1:
        raise AssertionError('Expected 16 unique local WPs and exactly one reference.')
    return frame


def finalize_local_scan__read_table(path, source, manifest):
    path = finalize_local_scan_Path(path)
    frame = finalize_local_scan_pd.read_csv(path, float_precision='round_trip')
    manifest.append(finalize_local_scan_common.candidate_input_record(path, source))
    for name in ('approved', 'purity_pass', 'valid_background', 'low_stat'):
        if name in frame and frame[name].dtype == object:
            values = frame[name].map({'True': True, 'False': False, True: True, False: False})
            if values.isna().any():
                raise ValueError(f'Invalid boolean field {name}: {path}')
            frame[name] = values.astype(bool)
    return frame


def finalize_local_scan__local_rows(frame, coordinates):
    return coordinates.merge(frame, on=['tx', 'ty'], how='inner', validate='one_to_many')


def finalize_local_scan__validate_cache_manifest(document, config, vr):
    expected = finalize_local_scan__vr_config(config, vr)
    settings = document['settings']
    fields = ('topology', 'vr', 'candidate_order', 'current_hist', 'expected_current_channel',
              'mass_min', 'mass_max', 'partner_iso_low', 'partner_iso_high',
              'tx', 'ty', 'scan_mu_iso', 'scan_abs_dphi', 'purity_limit', 'min_neff')
    for field in fields:
        if settings[field] != expected[field]:
            raise ValueError(f'{vr} cached configuration differs: {field}')
    if finalize_local_scan_Path(settings['bkg_dir']).resolve() != finalize_local_scan_Path(config['bkg_dir']).resolve():
        raise ValueError(f'{vr} cache refers to another background-MC production.')


def finalize_local_scan_load_cached_tables(config):
    """Read existing gated tables; no original DATA or signal file is opened."""
    finalize_local_scan__check_local_config(config)
    manifest, grids, purity, productions, validations = [], [], [], [], []
    for vr, directory_key in (('VR1', 'vr1_output_dir'), ('VR2', 'vr2_output_dir')):
        directory = finalize_local_scan_Path(config[directory_key])
        document_path = directory / 'manifest.json'
        finalize_local_scan__validate_cache_manifest(finalize_local_scan_json.loads(document_path.read_text()), config, vr)
        manifest.append(finalize_local_scan_common.candidate_input_record(document_path, f'{vr} cache configuration'))
        grid = finalize_local_scan__read_table(directory / 'grid.csv', f'{vr} gated DATA and MC grid', manifest)
        region_purity = finalize_local_scan__read_table(directory / 'purity_grid_regions.csv',
                                    f'{vr} 120-hypothesis purity grid', manifest)
        production = finalize_local_scan__read_table(directory / 'input_manifest.csv', f'{vr} production provenance', manifest)
        productions.append(production.loc[production.source.isin(['Background', 'Signal hypothesis'])]
                            .sort_values(['source', 'path']).reset_index(drop=True))
        grid['scope'] = vr + '_' + grid.plane
        region_purity['scope'] = vr + '_' + region_purity.plane
        for key in finalize_local_scan_expected_scopes(vr):
            part = grid.loc[grid.scope.eq(key)]
            if len(part) != 308 or part.duplicated(['source', 'tx', 'ty']).any():
                raise AssertionError(f'{key}: expected 154 unique MC and DATA grid rows.')
            if set(part.source) != {'MC', 'DATA'}:
                raise AssertionError(f'{key}: unexpected source labels.')
            data = part.loc[part.source.eq('DATA')]
            blocked = data.loc[~data.approved]
            if not blocked[finalize_local_scan_STATISTICS].isna().all().all():
                raise AssertionError(f'{key}: cached blocked DATA values were exposed.')
            validations.append(dict(scope=key, check='Unique 154-point gated MC/DATA cache; failed DATA hidden', passed=True))
        grids.append(grid)
        purity.append(region_purity)
    finalize_local_scan_pd.testing.assert_frame_equal(productions[0], productions[1])
    if productions[0].source.eq('Background').sum() != 18 or productions[0].source.eq('Signal hypothesis').sum() != 120:
        raise AssertionError('Cached production manifests require the same 18 backgrounds and 120 signals.')
    directory = finalize_local_scan_Path(config['vr2_output_dir'])
    background = finalize_local_scan__read_table(directory / 'sr_reference_grid.csv', 'Nominal SR MC grid', manifest)
    signals = finalize_local_scan__read_table(directory / 'sr_reference_signal_points.csv', 'Nominal SR signal grid; 120 hypotheses', manifest)
    sr_purity = finalize_local_scan__read_table(directory / 'sr_reference_purity_grid_regions.csv', 'Nominal SR 120-hypothesis purity grid', manifest)
    if set(background.source) != {'MC'} or set(background.context) != {'SR'} or len(background) != 154:
        raise AssertionError('Nominal SR cache must contain MC only and 154 WPs.')
    if len(signals) != 154 * 120 or signals['sample'].nunique() != 120:
        raise AssertionError('Nominal SR signals require 120 distinct hypotheses over all 154 WPs.')
    if signals.duplicated(['sample', 'tx', 'ty']).any():
        raise AssertionError('Duplicate signal hypothesis/grid coordinates.')
    background['scope'], sr_purity['scope'] = 'SR', 'SR'
    full_closure = finalize_local_scan_pd.concat([background, *grids], ignore_index=True)
    full_purity = finalize_local_scan_pd.concat([sr_purity, *purity], ignore_index=True)
    if set(full_closure.scope) != set(finalize_local_scan_SCOPE_ORDER) or set(full_purity.scope) != set(finalize_local_scan_SCOPE_ORDER):
        raise AssertionError('Expected nominal MC SR and exactly seven disjoint VR scopes.')
    validations.append(dict(scope='All', check='Identical 18-background/120-signal production in both VR caches', passed=True))
    validations.append(dict(scope='SR', check='Only MC SR cache; 120 separate signal hypotheses; no nominal DATA', passed=True))
    return dict(full_closure=full_closure, full_purity=full_purity,
                background=background, signals=signals, production=productions[0],
                input_manifest=manifest, validation=validations)


def finalize_local_scan_expected_scopes(vr):
    states = ('DD', 'PP', 'DP', 'PD') if vr == 'VR1' else ('PP', 'DP', 'PD')
    return tuple(vr + '_' + state for state in states)


def finalize_local_scan__load_background_planes(config, cached):
    """Project all eight scopes in one pass through the background production."""
    files = sorted(finalize_local_scan_Path(config['bkg_dir']).glob('*.coffea'))
    if len(files) != 18:
        raise ValueError('Expected exactly 18 original background-MC files.')
    planes = {scope: None for scope in finalize_local_scan_SCOPE_ORDER}
    samples = {scope: {} for scope in finalize_local_scan_SCOPE_ORDER}
    provenance = cached['production'].loc[cached['production'].source.eq('Background')]
    configs = {vr: finalize_local_scan__vr_config(config, vr) for vr in ('VR1', 'VR2')}
    specs = {vr: finalize_local_scan_common.candidate_specs(vr_config)[0] for vr, vr_config in configs.items()}
    manifest = []
    for index, path in enumerate(files, 1):
        record = finalize_local_scan_common.candidate_input_record(path, 'Original background MC')
        saved = provenance.loc[provenance.path.map(lambda value: finalize_local_scan_Path(value).resolve()).eq(path.resolve())]
        if len(saved) != 1 or saved.iloc[0].sha256 != record['sha256']:
            raise ValueError(f'Background-MC provenance differs from the cached scans: {path.name}')
        loaded, payload = finalize_local_scan_common._comparison_payload(path, 'sixd')
        weights, variances, edges = finalize_local_scan_common._comparison_current_cube(payload, path, config)
        incoming = {'SR': finalize_local_scan_common.project_mc_sr(weights, variances, edges, config)}
        for vr in ('VR1', 'VR2'):
            for key, spec in specs[vr].items():
                incoming[vr + '_' + key] = finalize_local_scan_common.project_candidate(weights, variances, edges, spec, configs[vr])
        for scope, plane in incoming.items():
            samples[scope][path.stem] = plane
            planes[scope] = finalize_local_scan_common.add_candidate(planes[scope], plane, finalize_local_scan_common._comparison_group(path.stem))
        manifest.append(record)
        del loaded, payload, weights, variances, incoming
        finalize_local_scan_gc.collect()
        print(f'Background MC {index}/{len(files)}: {path.stem}', flush=True)
    return planes, samples, manifest


def finalize_local_scan__validate_projected_planes(planes, cached, config):
    validations = []
    for scope, plane in planes.items():
        original = cached['full_closure'].loc[cached['full_closure'].scope.eq(scope)
                                               & cached['full_closure'].source.eq('MC')]
        for row in original.to_dict('records'):
            calculated = finalize_local_scan_common.plane_statistics(plane, row['tx'], row['ty'], min_neff=config['min_neff'])
            finalize_local_scan_np.testing.assert_allclose([calculated[key] for key in finalize_local_scan_STATISTICS],
                                       [row[key] for key in finalize_local_scan_STATISTICS], equal_nan=True,
                                       rtol=1e-9, atol=1e-8, err_msg=f'{scope}: {row["tx"]}, {row["ty"]}')
        for tx in config['local_iso']:
            finalize_local_scan_common._comparison_split(plane['xedges'], tx)
        for ty in config['local_dphi']:
            finalize_local_scan_common._comparison_split(plane['yedges'], ty)
        validations.append(dict(scope=scope, check='Direct MC projection reproduces all 154 cached ABCD and sumw2 points', passed=True))
    return validations


def finalize_local_scan__composition_tables(planes, sample_planes, coordinates):
    grouped, individual, dominance = [], [], []
    for point in coordinates.to_dict('records'):
        selected = {key: point[key] for key in ('wp', 'label', 'tx', 'ty')}
        for scope, plane in planes.items():
            totals = finalize_local_scan_common._comparison_regions(plane, point['tx'], point['ty'])
            for collection, source in ((grouped, plane['components']), (individual, sample_planes[scope])):
                for component, values in source.items():
                    part = finalize_local_scan_common._comparison_regions(dict(plane, **values), point['tx'], point['ty'])
                    for region, entry in part.items():
                        total = totals[region]['value']
                        collection.append(dict(**selected, scope=scope, region=region, component=component,
                                               **{'yield': entry['value']}, error=entry['error'], neff=entry['neff'],
                                               fraction=entry['value'] / total if total > 0 else finalize_local_scan_np.nan,
                                               total_yield=total))
    composition, sample_composition = finalize_local_scan_pd.DataFrame(grouped), finalize_local_scan_pd.DataFrame(individual)
    for (wp, scope, region), part in sample_composition.groupby(['wp', 'scope', 'region'], sort=False):
        total = part.total_yield.iloc[0]
        largest = part.loc[part['yield'].abs().idxmax()]
        groups = composition.loc[composition.wp.eq(wp) & composition.scope.eq(scope) & composition.region.eq(region)]
        dominant = groups.loc[groups['yield'].abs().idxmax()]
        dominance.append(dict(wp=wp, scope=scope, region=region, tx=largest.tx, ty=largest.ty,
                              label=largest.label, total_yield=total,
                              largest_sample=largest.component, largest_sample_yield=largest['yield'],
                              largest_sample_fraction=largest.fraction, largest_sample_neff=largest.neff,
                              dominant_component=dominant.component, dominant_component_fraction=dominant.fraction,
                              nonnegative_components=bool((groups['yield'] >= 0).all()),
                              diagnostic='MC central-value support; sample dominance is not an event-count estimate'))
    return composition, sample_composition, finalize_local_scan_pd.DataFrame(dominance)


def finalize_local_scan__composition_transfer(composition):
    records = []
    for (wp, region), part in composition.groupby(['wp', 'region'], sort=False):
        reference = part.loc[part.scope.eq('SR')].set_index('component')
        for scope in finalize_local_scan_SCOPE_ORDER[1:]:
            test = part.loc[part.scope.eq(scope)].set_index('component')
            first, second = reference.fraction.align(test.fraction, fill_value=0.)
            valid = (finalize_local_scan_np.isfinite(first).all() and finalize_local_scan_np.isfinite(second).all()
                     and (first >= 0).all() and (second >= 0).all())
            row = test.iloc[0]
            records.append(dict(wp=wp, label=row.label, tx=row.tx, ty=row.ty, scope=scope, region=region,
                                total_variation=.5 * float((first - second).abs().sum()) if valid else finalize_local_scan_np.nan,
                                sr_dominant=first.idxmax() if valid else '',
                                vr_dominant=second.idxmax() if valid else '', valid_composition=bool(valid),
                                interpretation='Composition distance only; no VR-to-SR correction or transfer is assumed'))
    return finalize_local_scan_pd.DataFrame(records)


def finalize_local_scan_run_local_scan(config):
    """Build the sixteen-WP summary and the independently recalculated MC planes."""
    cached = finalize_local_scan_load_cached_tables(config)
    coordinates = finalize_local_scan__coordinates(config)
    significance_validation = finalize_local_scan_review.validate_significance()
    full_significance = finalize_local_scan_review.signal_significance('2mu2e', cached['background'], cached['signals'])
    significance = finalize_local_scan__local_rows(full_significance, coordinates)
    summaries = finalize_local_scan_review.summarize_significance(significance)
    primary = summaries.loc[summaries.signal_set.eq('matching_decay')].copy()
    ranking = coordinates.merge(primary, on=['tx', 'ty'], validate='one_to_one')
    ranking = ranking.sort_values(['z_known_b_ratio_median', 'tx', 'ty'], ascending=[False, True, True]).reset_index(drop=True)
    ranking['rank'] = finalize_local_scan_np.arange(1, len(ranking) + 1)
    ranking['is_top5'] = ranking['rank'] <= config['top_n']
    if len(ranking) != 16 or not ranking.hypotheses.eq(60).all() or not ranking.z_known_b_ratio_valid.eq(60).all():
        raise AssertionError('Every local WP needs the same 60 valid matching-decay reference ratios.')
    coordinates = coordinates.merge(ranking[['wp', 'rank', 'is_top5']], on='wp', validate='one_to_one')
    closure = finalize_local_scan__local_rows(cached['full_closure'], coordinates)
    purity = finalize_local_scan__local_rows(cached['full_purity'], coordinates)
    planes, samples, background_manifest = finalize_local_scan__load_background_planes(config, cached)
    validations = cached['validation'] + finalize_local_scan__validate_projected_planes(planes, cached, config)
    composition, sample_composition, dominance = finalize_local_scan__composition_tables(planes, samples, coordinates)
    for (wp, scope, region), part in composition.groupby(['wp', 'scope', 'region']):
        total = closure.loc[closure.wp.eq(wp) & closure.scope.eq(scope) & closure.source.eq('MC'), region].iloc[0]
        finalize_local_scan_np.testing.assert_allclose(part['yield'].sum(), total, rtol=1e-10, atol=1e-8)
        finalize_local_scan_np.testing.assert_allclose((part.error ** 2).sum(), closure.loc[
            closure.wp.eq(wp) & closure.scope.eq(scope) & closure.source.eq('MC'), 'var_' + region].iloc[0],
            rtol=1e-10, atol=1e-8)
    validations.extend([dict(scope='All', check='MC group yields and sumw2 conserve every selected ABCD region', passed=True),
                        dict(scope='All', check='16 unique native WPs; top five selected by nominal MC only', passed=True)])
    validations.extend(dict(scope='SR', check=name, passed=bool(passed))
                       for name, passed in significance_validation.items())
    tables = dict(
        working_points=coordinates, ranking=ranking,
        significance_summary_all_sets=coordinates.merge(summaries, on=['tx', 'ty'], validate='one_to_many'),
        signal_significance=significance,
        signal_grid=finalize_local_scan__local_rows(cached['signals'], coordinates),
        background_grid=finalize_local_scan__local_rows(cached['background'], coordinates),
        closure=closure, full_closure=cached['full_closure'],
        purity=purity, full_purity=cached['full_purity'],
        composition=composition, sample_composition=sample_composition,
        dominance_summary=dominance, transfer_summary=finalize_local_scan__composition_transfer(composition),
        validation=finalize_local_scan_pd.DataFrame(validations),
        input_manifest=finalize_local_scan_pd.DataFrame(cached['input_manifest'] + background_manifest))
    metadata = dict(
        topology='2mu2e', local_working_points=16, scope_order=list(finalize_local_scan_SCOPE_ORDER), primary_signal_hypotheses=60,
        purity_signal_hypotheses=120, original_background_files_opened=18,
        original_signal_files_opened=0, original_data_files_opened=0,
        nominal_data_loaded=False, nominal_data_A_evaluated=False,
        ranking_uses_vr_data=False, background_provenance_verified=True,
        nominal_selection='DD; Mjj>=150 GeV; 0<=partner isolation<0.5; A: iso<tx and |dphi|>=ty',
        vr1_selection='0<=Mjj<150 GeV; DD/PP/DP/PD; 0<=partner isolation<0.5',
        vr2_selection='150<=Mjj<300 GeV; PP/DP/PD; 0<=partner isolation<0.5',
        primary_metric='Median of separate known-background Asimov Z(WP)/Z(reference) for 60 matching-decay hypotheses',
        data_gate='Existing per-WP all-120/all-ABCD S/B_MC<0.10 gate; blocked DATA stay NaN',
        limitations=[
            'MC central-value composition fractions can be uncertain at low effective counts.',
            'Composition distance is descriptive; it does not establish a VR-to-SR transfer factor.',
            'No VR closure correction is applied to the nominal SR.',
            'The known-background and MC-statistical counting metrics are not a final ABCD discovery likelihood.',
            'Nearby WPs share events and must be varied together in ranking diagnostics.',
            'Only native histogram boundaries are used; no interpolation or background floor is applied.'])
    return dict(config=config, tables=tables, planes=planes, sample_planes=samples, metadata=metadata)


def finalize_local_scan_export_local_scan(result, config, figures=None):
    """Export new numerical tables and provenance; the notebook retains the figures."""
    output = finalize_local_scan_Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in result['tables'].items():
        if isinstance(frame, finalize_local_scan_pd.DataFrame):
            frame.to_csv(output / (name + '.csv'), index=False)
    manifest_rows = [dict(order=index, figure=name, storage='Embedded in notebook')
                     for index, (_, name) in enumerate(figures or [], 1)]
    if manifest_rows:
        finalize_local_scan_pd.DataFrame(manifest_rows).to_csv(output / 'figure_manifest.csv', index=False)
    implementation = [finalize_local_scan_Path(__file__).with_name(name)
                      for name in ('abcd_finalize_common.py', 'abcd_finalize_plots.py')]
    manifest = dict(created_utc=finalize_local_scan_datetime.now(finalize_local_scan_timezone.utc).isoformat(), configuration=config,
                    methodology=result['metadata'],
                    implementation_hashes={path.name: finalize_local_scan_common._comparison_hash(path) for path in implementation},
                    table_rows={name: len(frame) for name, frame in result['tables'].items()
                                if isinstance(frame, finalize_local_scan_pd.DataFrame)}, figures=manifest_rows)
    (output / 'manifest.json').write_text(finalize_local_scan_json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + '\n')
    print(f'Saved numerical tables and provenance: {output}', flush=True)
    return finalize_local_scan_pd.DataFrame(manifest_rows)


def finalize_local_scan_build_vr2_focus_config(notebook_dir, selection_mode, force_reference=False, topology='2mu2e'):
    """Configure five ranked VR2 WPs; reference inclusion is optional and explicit."""
    if selection_mode not in ('sr_significance', 'pp_pd_closure'):
        raise ValueError('Choose selection_mode explicitly: sr_significance or pp_pd_closure.')
    if topology not in ('2mu2e', '4mu'):
        raise ValueError('Choose topology explicitly: 2mu2e or 4mu.')
    config = finalize_local_scan_build_config(notebook_dir)
    if topology == '4mu':
        directory = finalize_local_scan_Path(notebook_dir).resolve()
        config.update(
            topology=topology, current_hist='abcd6d_4mu',
            expected_current_channel='abcd6d_inclusive_4mu_dxySpread_cosAlpha_mu_veto_eHEM',
            notebook_path=directory / '08_MuIso_DPhi_4mu_LocalScan_summary.ipynb',
            vr1_output_dir=directory / '02_4mu_VR1_summary_outputs',
            vr2_output_dir=directory / '04_4mu_VR2_summary_outputs',
            sr_reference_dir=directory.parent / 'ABCD_plane_summary' / '12_Optimized_ABCD_4mu_comparison_outputs')
    notebook_number = '07' if topology == '2mu2e' else '08'
    config.update(
        selection_mode=selection_mode, force_reference=bool(force_reference),
        output_dir=finalize_local_scan_Path(notebook_dir).resolve() / f'{notebook_number}_{topology}_VR2_focused_outputs',
        scope_order=['VR2_PP', 'VR2_PD', 'VR2_DP'], display_count=5,
        evaluation_scopes=['VR2_PP', 'VR2_PD'],
        scan_iso_limits=(0., 1.), scan_dphi_limits=(1., 3.),
        vr='VR2', candidate_order=['PP', 'DP', 'PD'],
        mass_min=150., mass_max=300.)
    return config


def finalize_local_scan_run_vr2_focus(config):
    """Compare five ranked WPs using only the matching notebook-03/04 caches.

    PP and PD define the joint validation diagnostic. DP is retained for display
    after selection and never contributes to either ranking rule. No original
    coffea, VR1 table, nominal DATA, toy, or likelihood calculation is accessed.
    """
    mode = config.get('selection_mode')
    if mode not in ('sr_significance', 'pp_pd_closure'):
        raise ValueError('An explicit sr_significance or pp_pd_closure ranking is required.')
    if (config['topology'] not in ('2mu2e', '4mu') or tuple(config['local_iso']) != finalize_local_scan_LOCAL_ISO
            or tuple(config['local_dphi']) != finalize_local_scan_LOCAL_DPHI
            or tuple(config['reference_wp']) != finalize_local_scan_REFERENCE_WP):
        raise ValueError('Keep the established channel, native local domain and reference.')
    if (config.get('display_count') != 5
            or list(config.get('scope_order', [])) != ['VR2_PP', 'VR2_PD', 'VR2_DP']
            or list(config.get('evaluation_scopes', [])) != ['VR2_PP', 'VR2_PD']):
        raise ValueError('Evaluate PP/PD and display exactly five WPs in PP/PD/DP.')
    finalize_local_scan_common.validate_finalize_config(config)
    source_notebook = '03' if config['topology'] == '2mu2e' else '04'
    directory = finalize_local_scan_Path(config['vr2_output_dir'])
    manifest, validation = [], []
    document_path = directory / 'manifest.json'
    document = finalize_local_scan_json.loads(document_path.read_text())
    finalize_local_scan__validate_cache_manifest(document, config, 'VR2')
    manifest.append(finalize_local_scan_common.candidate_input_record(document_path, f'Notebook {source_notebook} VR2 configuration'))
    full_closure = finalize_local_scan__read_table(directory / 'grid.csv', f'Notebook {source_notebook} gated VR2 MC/DATA grid', manifest)
    full_purity = finalize_local_scan__read_table(directory / 'purity_grid_regions.csv', f'Notebook {source_notebook} VR2 120-model purity grid', manifest)
    background = finalize_local_scan__read_table(directory / 'sr_reference_grid.csv', f'Notebook {source_notebook} nominal SR MC grid', manifest)
    signals = finalize_local_scan__read_table(directory / 'sr_reference_signal_points.csv', f'Notebook {source_notebook} nominal SR signal grid', manifest)
    production = finalize_local_scan__read_table(directory / 'input_manifest.csv', f'Notebook {source_notebook} production provenance', manifest)
    if (production.source.eq('Background').sum() != 18
            or production.source.eq('Signal hypothesis').sum() != 120):
        raise AssertionError('The cached production requires 18 backgrounds and 120 signal hypotheses.')
    if (set(full_closure.plane) != {'PP', 'PD', 'DP'}
            or set(full_closure.source) != {'MC', 'DATA'}
            or set(full_closure.context) != {'VR'}):
        raise AssertionError('The full closure table must contain only VR2 PP/PD/DP MC and DATA.')
    full_closure['scope'] = 'VR2_' + full_closure.plane
    full_purity['scope'] = 'VR2_' + full_purity.plane
    native = [(float(x), float(y)) for x in config['scan_mu_iso'] for y in config['scan_abs_dphi']]
    expected_coordinates = set(native)
    for scope in config['scope_order']:
        for source in ('MC', 'DATA'):
            frame = full_closure.loc[full_closure.scope.eq(scope) & full_closure.source.eq(source)]
            if (len(frame) != 154 or frame.duplicated(['tx', 'ty']).any()
                    or set(zip(frame.tx, frame.ty)) != expected_coordinates):
                raise AssertionError(f'{scope}/{source}: expected the complete 154-point native grid.')
            if source == 'DATA':
                blocked = frame.loc[~frame.approved]
                if not blocked[finalize_local_scan_STATISTICS].isna().all().all():
                    raise AssertionError(f'{scope}: blocked DATA values must remain hidden.')
        part = full_purity.loc[full_purity.scope.eq(scope)]
        if (len(part) != 154 * 4 or part.duplicated(['tx', 'ty', 'region']).any()
                or set(part.region) != set('ABCD')):
            raise AssertionError(f'{scope}: expected a four-region purity record at every native WP.')
        validation.append(dict(scope=scope, check='Complete 154-WP VR2 grid; blocked DATA remain NaN', passed=True))
    if (len(full_closure) != 924 or len(full_purity) != 1848
            or set(full_purity.scope) != set(config['scope_order'])):
        raise AssertionError('Only the three VR2 categories may appear in the focused tables.')
    if (len(background) != 154 or set(background.source) != {'MC'}
            or set(background.context) != {'SR'}
            or len(signals) != 154 * 120 or signals['sample'].nunique() != 120):
        raise AssertionError('The SR inputs must contain 154 MC points and 120 separate signal hypotheses.')

    coordinates = finalize_local_scan__coordinates(config).rename(columns={'wp': 'source_id'})
    full_significance = finalize_local_scan_review.signal_significance(config['topology'], background, signals)
    local_significance = coordinates.merge(full_significance, on=['tx', 'ty'], validate='one_to_many')
    summaries = finalize_local_scan_review.summarize_significance(local_significance)
    primary = summaries.loc[summaries.signal_set.eq('matching_decay')].copy()
    pool = coordinates.merge(primary, on=['tx', 'ty'], validate='one_to_one')
    if (len(pool) != 16 or not pool.hypotheses.eq(60).all()
            or not pool.z_known_b_ratio_valid.eq(60).all()):
        raise AssertionError('Every local candidate requires the same 60 valid matching-decay ratios.')
    for plane, prefix in (('PP', 'pp'), ('PD', 'pd')):
        data = full_closure.loc[full_closure.plane.eq(plane) & full_closure.source.eq('DATA')].copy()
        data['min_count'] = data[list('ABCD')].min(axis=1)
        columns = ['tx', 'ty', 'kappa', 'kappa_error', 'approved', 'valid_background',
                   'min_count', 'max_S_B', 'low_stat']
        renamed = data[columns].rename(columns={name: prefix + '_' + name for name in columns[2:]})
        pool = pool.merge(renamed, on=['tx', 'ty'], validate='one_to_one')
        valid = (pool[prefix + '_approved'] & pool[prefix + '_valid_background']
                 & finalize_local_scan_np.isfinite(pool[prefix + '_kappa'])
                 & finalize_local_scan_np.isfinite(pool[prefix + '_kappa_error'])
                 & pool[prefix + '_kappa_error'].gt(0))
        pool[prefix + '_evaluation_valid'] = valid
    pool['joint_closure_eligible'] = pool.pp_evaluation_valid & pool.pd_evaluation_valid
    pool['pp_pd_closure_score'] = finalize_local_scan_np.where(
        pool.joint_closure_eligible,
        ((pool.pp_kappa - 1.) / pool.pp_kappa_error) ** 2
        + ((pool.pd_kappa - 1.) / pool.pd_kappa_error) ** 2, finalize_local_scan_np.nan)
    pool['mode'] = mode
    sr_order = pool.sort_values(['z_known_b_ratio_median', 'tx', 'ty'], ascending=[False, True, True])
    pool['sr_rank'] = pool.source_id.map(dict(zip(sr_order.source_id, range(1, 17))))
    closure_order = pool.sort_values(['pp_pd_closure_score', 'tx', 'ty'], na_position='last')
    pool['closure_rank'] = pool.source_id.map(dict(zip(closure_order.source_id, range(1, 17))))
    eligible = pool.loc[pool.joint_closure_eligible]
    if mode == 'sr_significance':
        ordered = pool.sort_values(['z_known_b_ratio_median', 'tx', 'ty'], ascending=[False, True, True])
    else:
        ordered = eligible.sort_values(['pp_pd_closure_score', 'tx', 'ty'])
    if len(ordered) < 5 or ordered.is_reference.sum() != 1:
        raise AssertionError('Expected at least five ranked candidates and one normalization reference.')
    pool['selection_rank'] = pool.source_id.map(dict(zip(ordered.source_id, range(1, len(ordered) + 1))))
    if bool(config.get('force_reference', False)):
        choices = ordered.loc[~ordered.is_reference].head(4)
        reference = ordered.loc[ordered.is_reference]
        chosen = finalize_local_scan_pd.concat([choices, reference], ignore_index=True)
    else:
        chosen = ordered.head(5).reset_index(drop=True)
    selected = chosen[['source_id', 'tx', 'ty', 'label', 'is_reference']].copy()
    selected.insert(0, 'wp', ['Ref' if reference else f'WP{index}'
                             for index, reference in enumerate(selected.is_reference, 1)])
    selected['rank'] = finalize_local_scan_np.arange(1, 6)
    selected['is_top5'] = (chosen.source_id.map(pool.set_index('source_id').selection_rank) <= 5).to_numpy()
    ranking = selected.merge(pool.drop(columns=['tx', 'ty', 'label', 'is_reference']),
                             on='source_id', validate='one_to_one')
    closure = selected.merge(full_closure, on=['tx', 'ty'], validate='one_to_many')
    purity = selected.merge(full_purity, on=['tx', 'ty'], validate='one_to_many')
    selected_signals = selected.merge(signals, on=['tx', 'ty'], validate='one_to_many')
    selected_significance = selected.merge(full_significance, on=['tx', 'ty'], validate='one_to_many')
    selected_background = selected.merge(background, on=['tx', 'ty'], validate='one_to_one')
    if (len(selected) != 5 or selected.is_reference.sum() > 1
            or selected.duplicated(['tx', 'ty']).any()
            or len(closure) != 30 or len(purity) != 60
            or len(selected_signals) != 600 or len(selected_significance) != 600):
        raise AssertionError('The focused outputs require exactly five WPs, three VRs, and 120 signals per WP.')
    if not bool(config.get('force_reference', False)):
        expected_reference = bool(ordered.iloc[:5].is_reference.any())
        if bool(selected.is_reference.any()) != expected_reference or not selected.is_top5.all():
            raise AssertionError('Without forced inclusion, the reference appears only if it is genuinely in the top five.')
    for source in ('MC', 'DATA'):
        if closure.loc[closure.source.eq(source)].groupby('wp').size().ne(3).any():
            raise AssertionError('Each displayed WP requires PP, PD, and DP for both sources.')
    validation.extend([
        dict(scope='VR2', check=f'Notebook-{source_notebook} cuts, histogram identity, native boundaries and production preserved', passed=True),
        dict(scope='Local', check='16 unique native candidates; identical 60-model significance cohort', passed=True),
        dict(scope='Displayed', check='Exactly five ranked WPs; reference is not a forced sixth point', passed=True),
        dict(scope='Evaluation', check='PP and PD only; DP excluded from both selection rules', passed=True),
        dict(scope='Evaluation', check='Selected PP/PD DATA retain the existing 120-model gate and masking policy', passed=True),
        dict(scope='Inputs', check='Cached tables only; no raw MC/signal/DATA or VR1 inputs opened', passed=True),
        dict(scope='SR', check='Nominal SR input is MC only; no nominal DATA access', passed=True),
    ])
    validation.extend(dict(scope='SR', check=name, passed=bool(passed))
                      for name, passed in finalize_local_scan_review.validate_significance().items())
    summary_columns = ['wp', 'source_id', 'tx', 'ty', 'is_reference',
                       'z_known_b_ratio_median', 'z_known_b_ratio_q10', 'z_known_b_ratio_q90',
                       'b', 'sigma_b', 'neff_b', 'pp_kappa', 'pp_kappa_error',
                       'pd_kappa', 'pd_kappa_error', 'pp_pd_closure_score',
                       'pp_min_count', 'pd_min_count', 'mode']
    tables = dict(
        working_points=selected, ranking=ranking, summary=ranking[summary_columns].copy(),
        candidate_pool=pool.sort_values('selection_rank', na_position='last').reset_index(drop=True),
        full_closure=full_closure, full_purity=full_purity, closure=closure, purity=purity,
        signal_grid=selected_signals, signal_significance=selected_significance,
        background_grid=selected_background,
        validation=finalize_local_scan_pd.DataFrame(validation), input_manifest=finalize_local_scan_pd.DataFrame(manifest))
    metadata = dict(
        topology=config['topology'], scope_order=list(config['scope_order']),
        evaluation_scopes=['VR2_PP', 'VR2_PD'], selection_mode=mode,
        local_candidate_count=16, displayed_working_points=5,
        reference_included=bool(selected.is_reference.any()),
        reference_forced=bool(config.get('force_reference', False)),
        reference_used_for_normalization=True,
        primary_signal_hypotheses=60, contamination_signal_hypotheses=120,
        ranking_uses_vr_data=mode == 'pp_pd_closure', dp_used_for_ranking=False,
        original_background_files_opened=0, original_signal_files_opened=0,
        original_data_files_opened=0, nominal_data_loaded=False, vr1_inputs_loaded=False,
        cached_source=f'{source_notebook}_{config["topology"]}_VR2_summary_outputs only',
        vr2_selection='150<=Mjj<300 GeV; PP/PD/DP; 0<=partner isolation<0.5',
        nominal_selection='MC-only DD; Mjj>=150 GeV; 0<=partner isolation<0.5',
        primary_metric=('Median per-hypothesis nominal-SR MC Z ratio for 60 separate '
                        + ('2Mu2E' if config['topology'] == '2mu2e' else '4Mu') + ' models'),
        closure_metric='Sum over PP and PD of ((kappa-1)/sigma_kappa)^2; descriptive score, not a calibrated p-value',
        data_gate='Existing all-120/all-ABCD S/B_MC<0.10 gate and valid positive MC support',
        normalization='Unchanged production signal/background normalization',
        limitations=[
            'The sixteen-WP candidate pool is internal; only five ranked WPs are displayed.',
            'The normalization reference is displayed only when selected, unless explicitly forced.',
            'The closure score is descriptive and may favor larger uncertainties; correlated systematics are not modeled.',
            'Nearby WP scores reuse events and are correlated.',
            'No VR-to-SR kappa correction or uncertainty is adopted.',
            'MC counting significance is an exploratory sensitivity metric, not a final ABCD discovery claim.'])
    return dict(config=config, tables=tables, metadata=metadata)


def finalize_local_scan_vr2_focus_summary_tables(result):
    """Pack complete selected-WP metrics into five readable, multiline blocks.

    The companion numeric table retains every exported value without display
    rounding. Signal medians use 60 separate matching hypotheses; purity maxima
    use the existing all-120/all-ABCD contamination diagnostic.
    """
    points = result['tables']['working_points'].sort_values('rank')
    ranking = result['tables']['ranking'].set_index('wp')
    signal = result['tables']['signal_significance']
    signal = signal.loc[signal.matching_decay]
    background = result['tables']['background_grid'].set_index('wp')
    closure = result['tables']['closure']
    display_rows, numeric_rows = [], []
    for point in points.itertuples():
        metric = ranking.loc[point.wp]
        bg = background.loc[point.wp]
        models = signal.loc[signal.wp.eq(point.wp)]
        if len(models) != 60:
            raise AssertionError('Each summary row requires 60 matching signal hypotheses.')
        formatted = {'WP / cuts': f'{point.wp}\niso = {point.tx:.2f}\n|Δφ| = {point.ty:.1f}'}
        formatted['Nominal SR MC'] = (
            f'S(A), model median: {models.s.median():.4g}\n'
            f'B(A): {metric.b:.3f} ± {metric.sigma_b:.3f}\n'
            f'N_eff(A): {metric.neff_b:.2f}; min CR: {bg.min_sideband_neff:.2f}\n'
            f'κ: {bg.kappa:.3f} ± {bg.kappa_error:.3f}\n'
            f'Z, model median: {metric.z_known_b_median:.5f}\n'
            f'Z/ref: {metric.z_known_b_ratio_median:.3f} '
            f'[{metric.z_known_b_ratio_q10:.3f}, {metric.z_known_b_ratio_q90:.3f}]\n'
            f'MC-stat diagnostic Z/ref: {metric.z_mc_stat_diagnostic_ratio_median:.3f}')
        numeric = dict(wp=point.wp, tx=point.tx, ty=point.ty,
                       sr_signal_A_median=float(models.s.median()),
                       sr_background_A=float(metric.b), sr_background_error=float(metric.sigma_b),
                       sr_neff_A=float(metric.neff_b), sr_min_sideband_neff=float(bg.min_sideband_neff),
                       sr_kappa=float(bg.kappa), sr_kappa_error=float(bg.kappa_error),
                       z_median=float(metric.z_known_b_median),
                       z_q10=float(models.z_known_b.quantile(.1)),
                       z_q90=float(models.z_known_b.quantile(.9)),
                       z_ratio_median=float(metric.z_known_b_ratio_median),
                       z_ratio_q10=float(metric.z_known_b_ratio_q10),
                       z_ratio_q90=float(metric.z_known_b_ratio_q90),
                       z_mc_stat_ratio_median=float(metric.z_mc_stat_diagnostic_ratio_median))
        for category in ('PP', 'PD', 'DP'):
            part = closure.loc[closure.wp.eq(point.wp) & closure.scope.eq('VR2_' + category)]
            data = part.loc[part.source.eq('DATA')].iloc[0]
            mc = part.loc[part.source.eq('MC')].iloc[0]
            allowed = bool(data.approved)
            if allowed:
                counts = '/'.join(f'{data[r]:.0f}' for r in 'ABCD')
                data_lines = (f'DATA A/B/C/D: {counts}\n'
                              f'BC/D: {data.pred:.1f} ± {data.pred_error:.1f}\n'
                              f'κ DATA: {data.kappa:.3f} ± {data.kappa_error:.3f}')
                min_count = f'{min(data[r] for r in "ABCD"):.0f}'
            else:
                data_lines = 'DATA A/B/C/D: gated\nBC/D: hidden\nκ DATA: hidden'
                min_count = 'hidden'
            formatted[category] = (
                data_lines + f'\nκ MC: {mc.kappa:.3f} ± {mc.kappa_error:.3f}\n'
                f'max S/B: {100*mc.max_S_B:.3f}% | gate: {"PASS" if allowed else "BLOCKED"}\n'
                f'min DATA: {min_count}; MC N_eff,min: {mc.min_neff:.2f}')
            for key in list('ABCD') + ['pred', 'pred_error', 'kappa', 'kappa_error',
                                      'min_neff', 'min_sideband_neff', 'max_S_B', 'approved']:
                numeric[category + '_DATA_' + key] = data[key]
                numeric[category + '_MC_' + key] = mc[key]
        display_rows.append(formatted)
        numeric_rows.append(numeric)
    display_table, numeric_table = finalize_local_scan_pd.DataFrame(display_rows), finalize_local_scan_pd.DataFrame(numeric_rows)
    if len(display_table) != 5 or len(numeric_table) != 5:
        raise AssertionError('The comprehensive summary must contain exactly five WPs.')
    return dict(display=display_table, numeric=numeric_table)


# ========================================================================
# abcd_local_scan_statistics: original study API
# ========================================================================
"""Correlated MC diagnostics and conditional four-region Asimov calculations.

The resampling is a scaled-Poisson approximation to positive disjoint-bin
moments, not a raw-event bootstrap or an additional source of MC information.
The optional likelihood scenarios use the nominal MC background as an Asimov
truth. Their kappa constraint is explicitly supplied, never inferred from VR
DATA here. Neither operation reads nominal DATA.
"""
import numpy as finalize_local_scan_statistics_np
import pandas as finalize_local_scan_statistics_pd
from scipy.optimize import brentq as finalize_local_scan_statistics_brentq; from scipy.optimize import minimize as finalize_local_scan_statistics_minimize
from scipy.stats import rankdata as finalize_local_scan_statistics_rankdata

finalize_local_scan_statistics_asimov_significance = _FINALIZE_NAMESPACES['review_mc_sr_working_points'].asimov_significance


def finalize_local_scan_statistics__coordinate_values(frame, coordinates, column):
    selected = frame.copy()
    selected['tx'] = selected.tx.round(12)
    selected['ty'] = selected.ty.round(12)
    if selected.duplicated(['tx', 'ty']).any():
        raise ValueError('Each coordinate must have exactly one input row.')
    index = finalize_local_scan_statistics_pd.MultiIndex.from_arrays([coordinates.tx.round(12), coordinates.ty.round(12)])
    values = selected.set_index(['tx', 'ty'])[column].reindex(index).to_numpy(float)
    if not finalize_local_scan_statistics_np.isfinite(values).all():
        raise ValueError(f'Missing or nonfinite cumulative input: {column}')
    return values


def finalize_local_scan_statistics__local_partition(coordinates):
    xs = finalize_local_scan_statistics_np.sort(coordinates.tx.unique())
    ys = finalize_local_scan_statistics_np.sort(coordinates.ty.unique())
    if len(coordinates) != len(xs) * len(ys) or coordinates.duplicated(['tx', 'ty']).any():
        raise ValueError('The local coordinates must form a complete rectangular grid.')
    order = finalize_local_scan_statistics_pd.DataFrame([(x, y) for x in xs for y in ys], columns=['tx', 'ty'])
    masks = finalize_local_scan_statistics_np.asarray([[(ix <= finalize_local_scan_statistics_np.searchsorted(xs, row.tx)
                          and iy >= finalize_local_scan_statistics_np.searchsorted(ys, row.ty))
                         for ix in range(len(xs)) for iy in range(len(ys))]
                        for row in coordinates.itertuples()], dtype=float)
    return xs, ys, order, masks


def finalize_local_scan_statistics__cumulative_to_atoms(cumulative, shape):
    cumulative = finalize_local_scan_statistics_np.asarray(cumulative, dtype=float).reshape(shape)
    increments = finalize_local_scan_statistics_np.diff(finalize_local_scan_statistics_np.vstack([finalize_local_scan_statistics_np.zeros((1, shape[1])), cumulative]), axis=0)
    return (increments - finalize_local_scan_statistics_np.column_stack([increments[:, 1:], finalize_local_scan_statistics_np.zeros(shape[0])])).ravel()


def finalize_local_scan_statistics__positive_moments(means, variances):
    means, variances = finalize_local_scan_statistics_np.broadcast_arrays(finalize_local_scan_statistics_np.asarray(means, float), finalize_local_scan_statistics_np.asarray(variances, float))
    means, variances = means.copy(), variances.copy()
    tolerance_m = 128 * finalize_local_scan_statistics_np.finfo(float).eps * finalize_local_scan_statistics_np.maximum(
        finalize_local_scan_statistics_np.max(finalize_local_scan_statistics_np.abs(means), axis=-1, keepdims=True), finalize_local_scan_statistics_np.finfo(float).tiny)
    tolerance_v = 128 * finalize_local_scan_statistics_np.finfo(float).eps * finalize_local_scan_statistics_np.maximum(
        finalize_local_scan_statistics_np.max(finalize_local_scan_statistics_np.abs(variances), axis=-1, keepdims=True), finalize_local_scan_statistics_np.finfo(float).tiny)
    if (not finalize_local_scan_statistics_np.isfinite(means).all() or not finalize_local_scan_statistics_np.isfinite(variances).all()
            or finalize_local_scan_statistics_np.any(means < -tolerance_m) or finalize_local_scan_statistics_np.any(variances < -tolerance_v)):
        raise ValueError('Signed or invalid disjoint-bin moments are unsupported by this positive-bin diagnostic.')
    repaired = int(finalize_local_scan_statistics_np.sum(means < 0) + finalize_local_scan_statistics_np.sum(variances < 0))
    means[means < 0] = 0
    variances[variances < 0] = 0
    # A tiny residual second moment can follow subtraction of equal cumulants.
    empty_residual = (means == 0) & (variances > 0) & (variances <= tolerance_v)
    repaired += int(empty_residual.sum())
    variances[empty_residual] = 0
    if finalize_local_scan_statistics_np.any((means == 0) & (variances > 0)):
        raise ValueError('A zero-mean bin with positive variance needs signed-weight information.')
    return means, variances, repaired


def finalize_local_scan_statistics__moment_draws(means, variances, n_toys, rng):
    means, variances, repaired = finalize_local_scan_statistics__positive_moments(means, variances)
    draws = finalize_local_scan_statistics_np.broadcast_to(means, (n_toys,) + means.shape).copy()
    uncertain = (means > 0) & (variances > 0)
    effective = means[uncertain] ** 2 / variances[uncertain]
    scale = variances[uncertain] / means[uncertain]
    draws[:, uncertain] = rng.poisson(effective, size=(n_toys, int(uncertain.sum()))) * scale
    return draws, repaired


def finalize_local_scan_statistics__rank_frequencies(scores):
    if not len(scores):
        shape = scores.shape[1]
        return {name: finalize_local_scan_statistics_np.full(shape, finalize_local_scan_statistics_np.nan) for name in
                ('rank_median', 'rank_q10', 'rank_q90', 'p_top1', 'p_top5')}
    ranks = finalize_local_scan_statistics_rankdata(-scores, method='average', axis=1)
    lower = finalize_local_scan_statistics_rankdata(-scores, method='min', axis=1)
    upper = finalize_local_scan_statistics_rankdata(-scores, method='max', axis=1)
    multiplicity = upper - lower + 1
    return dict(rank_median=finalize_local_scan_statistics_np.median(ranks, axis=0),
                rank_q10=finalize_local_scan_statistics_np.quantile(ranks, .1, axis=0),
                rank_q90=finalize_local_scan_statistics_np.quantile(ranks, .9, axis=0),
                p_top1=finalize_local_scan_statistics_np.clip((2 - lower) / multiplicity, 0, 1).mean(axis=0),
                p_top5=finalize_local_scan_statistics_np.clip((6 - lower) / multiplicity, 0, 1).mean(axis=0))


def finalize_local_scan_statistics_local_rank_diagnostics(result, config):
    """Keep common disjoint-bin fluctuations when comparing overlapping WPs.

    Rank frequencies are conditional on all WPs having positive background and
    all 60 reference signal significances being positive in the same replica.
    Undefined replicas are counted explicitly rather than assigned a floor.
    """
    tables = result['tables']
    coordinates = tables['working_points'].sort_values('wp').reset_index(drop=True)
    xs, ys, order, masks = finalize_local_scan_statistics__local_partition(coordinates)
    shape = (len(xs), len(ys))
    background = tables['background_grid']
    mean_b = finalize_local_scan_statistics__cumulative_to_atoms(finalize_local_scan_statistics__coordinate_values(background, order, 'A'), shape)
    variance_b = finalize_local_scan_statistics__cumulative_to_atoms(finalize_local_scan_statistics__coordinate_values(background, order, 'var_A'), shape)
    signal = tables['signal_grid']
    samples = sorted(sample for sample in signal['sample'].unique() if sample.startswith('2Mu2E_'))
    if len(samples) != 60:
        raise ValueError('The primary ranking requires the same 60 separate 2Mu2E hypotheses.')
    mean_s = finalize_local_scan_statistics_np.asarray([finalize_local_scan_statistics__cumulative_to_atoms(finalize_local_scan_statistics__coordinate_values(
        signal.loc[signal['sample'].eq(sample)], order, 'S_A'), shape) for sample in samples])
    variance_s = finalize_local_scan_statistics_np.asarray([finalize_local_scan_statistics__cumulative_to_atoms(finalize_local_scan_statistics__coordinate_values(
        signal.loc[signal['sample'].eq(sample)], order, 'var_S_A'), shape) for sample in samples])
    reference = finalize_local_scan_statistics_np.flatnonzero(coordinates.is_reference.to_numpy(bool))
    if len(reference) != 1:
        raise ValueError('Exactly one local reference WP is required.')
    reference = int(reference[0])
    n_toys = int(config.get('n_toys', 2000))
    if n_toys < 1:
        raise ValueError('n_toys must be positive.')
    rng = finalize_local_scan_statistics_np.random.default_rng(int(config.get('rng_seed', 73021)))
    rows, diagnostics = [], []
    covariance_rows = []
    covariance = (masks * variance_b[None, :]) @ masks.T
    for i, a in coordinates.iterrows():
        for j, b in coordinates.iterrows():
            denominator = finalize_local_scan_statistics_np.sqrt(covariance[i, i] * covariance[j, j])
            covariance_rows.append(dict(wp_a=a.wp, wp_b=b.wp, covariance=covariance[i, j],
                                        correlation=covariance[i, j] / denominator if denominator > 0 else finalize_local_scan_statistics_np.nan,
                                        variance_a=covariance[i, i], variance_b=covariance[j, j]))
    try:
        background_draws, repaired_b = finalize_local_scan_statistics__moment_draws(mean_b, variance_b, n_toys, rng)
        signal_draws, repaired_s = finalize_local_scan_statistics__moment_draws(mean_s, variance_s, n_toys, rng)
    except ValueError as error:
        for model in ('background_only', 'background_and_signal'):
            diagnostics.append(dict(model=model, n_toys=n_toys, n_complete_toys=0,
                                    complete_fraction=0., invalid_reference_significance_fraction=finalize_local_scan_statistics_np.nan,
                                    any_nonpositive_background_fraction=finalize_local_scan_statistics_np.nan,
                                    status='UNSUPPORTED_SIGNED_MOMENTS', method=str(error)))
            for point in coordinates.to_dict('records'):
                rows.append(dict(point, model=model, rank_median=finalize_local_scan_statistics_np.nan, rank_q10=finalize_local_scan_statistics_np.nan,
                                 rank_q90=finalize_local_scan_statistics_np.nan, p_top1=finalize_local_scan_statistics_np.nan, p_top5=finalize_local_scan_statistics_np.nan,
                                 ratio_median=finalize_local_scan_statistics_np.nan, ratio_q10=finalize_local_scan_statistics_np.nan, ratio_q90=finalize_local_scan_statistics_np.nan,
                                 invalid_bg_fraction=finalize_local_scan_statistics_np.nan, n_complete_toys=0, n_toys=n_toys,
                                 status='UNSUPPORTED_SIGNED_MOMENTS'))
        return dict(ranking_toys=finalize_local_scan_statistics_pd.DataFrame(rows), rank_covariance=finalize_local_scan_statistics_pd.DataFrame(covariance_rows),
                    toy_diagnostics=finalize_local_scan_statistics_pd.DataFrame(diagnostics))
    b_draws = background_draws @ masks.T
    fixed_s = (mean_s @ masks.T).T
    varied_s = finalize_local_scan_statistics_np.transpose(signal_draws @ masks.T, (0, 2, 1))
    central_b = mean_b @ masks.T
    finalize_local_scan_statistics_np.testing.assert_allclose(central_b, finalize_local_scan_statistics__coordinate_values(background, coordinates, 'A'),
                               rtol=1e-9, atol=1e-8)
    finalize_local_scan_statistics_np.testing.assert_allclose(finalize_local_scan_statistics_np.diag(covariance),
                               finalize_local_scan_statistics__coordinate_values(background, coordinates, 'var_A'), rtol=1e-9, atol=1e-8)
    for model, s_draws in [('background_only', None), ('background_and_signal', varied_s)]:
        scores = finalize_local_scan_statistics_np.full((n_toys, len(coordinates)), finalize_local_scan_statistics_np.nan)
        complete = finalize_local_scan_statistics_np.zeros(n_toys, dtype=bool)
        invalid_reference = finalize_local_scan_statistics_np.zeros(n_toys, dtype=bool)
        for start in range(0, n_toys, 128):
            stop = min(start + 128, n_toys)
            signal_values = finalize_local_scan_statistics_np.broadcast_to(fixed_s, (stop - start,) + fixed_s.shape) if s_draws is None else s_draws[start:stop]
            z = finalize_local_scan_statistics_asimov_significance(signal_values, b_draws[start:stop, :, None])
            denominator = z[:, reference:reference + 1, :]
            ratios = finalize_local_scan_statistics_np.full_like(z, finalize_local_scan_statistics_np.nan)
            finalize_local_scan_statistics_np.divide(z, denominator, out=ratios, where=finalize_local_scan_statistics_np.isfinite(denominator) & (denominator > 0))
            complete[start:stop] = finalize_local_scan_statistics_np.isfinite(ratios).all(axis=(1, 2))
            invalid_reference[start:stop] = (~finalize_local_scan_statistics_np.isfinite(denominator) | (denominator <= 0)).any(axis=(1, 2))
            scores[start:stop] = finalize_local_scan_statistics_np.median(ratios, axis=2)
        selected = scores[complete]
        frequencies = finalize_local_scan_statistics__rank_frequencies(selected)
        quantiles = (finalize_local_scan_statistics_np.quantile(selected, [.1, .5, .9], axis=0) if len(selected)
                     else finalize_local_scan_statistics_np.full((3, len(coordinates)), finalize_local_scan_statistics_np.nan))
        for i, point in coordinates.iterrows():
            rows.append(dict(point, model=model, **{key: values[i] for key, values in frequencies.items()},
                             ratio_q10=quantiles[0, i], ratio_median=quantiles[1, i],
                             ratio_q90=quantiles[2, i], invalid_bg_fraction=float((b_draws[:, i] <= 0).mean()),
                             n_complete_toys=int(complete.sum()), n_toys=n_toys,
                             status='OK' if complete.any() else 'NO_COMPLETE_REPLICAS'))
        diagnostics.append(dict(model=model, n_toys=n_toys, n_complete_toys=int(complete.sum()),
                                complete_fraction=float(complete.mean()),
                                invalid_reference_significance_fraction=float(invalid_reference.mean()),
                                any_nonpositive_background_fraction=float((b_draws <= 0).any(axis=1).mean()),
                                roundoff_moment_repairs=repaired_b + repaired_s,
                                status='OK' if complete.any() else 'NO_COMPLETE_REPLICAS',
                                method='Positive disjoint-bin scaled-Poisson moment replicas; shared background; fixed 60-model cohort; conditional ranks, not confidence probabilities.'))
    return dict(ranking_toys=finalize_local_scan_statistics_pd.DataFrame(rows), rank_covariance=finalize_local_scan_statistics_pd.DataFrame(covariance_rows),
                toy_diagnostics=finalize_local_scan_statistics_pd.DataFrame(diagnostics))


def finalize_local_scan_statistics_asimov_abcd_significance(signal, background, kappa_log_width=0.):
    """Profile a four-region Poisson null with an optional log-kappa constraint.

    n_r = b_r + s_r is the signal-plus-background Asimov sample. The background
    truth fixes kappa0 = b_A*b_D/(b_B*b_C). Signal enters all four Asimov bins.
    Under mu=0, the free sideband means and log-kappa are profiled. The score
    equations reduce exactly to one strictly monotone scalar root in delta:
      lambda = n + delta*(1,-1,-1,1),
      log(lambda_A*lambda_D/(lambda_B*lambda_C)) - log(kappa0)
          + sigma_logkappa**2 * delta = 0.
    No MC statistical nuisance or VR-derived calibration is included. This is
    a conditional scenario, not the final discovery significance.
    """
    s, b = finalize_local_scan_statistics_np.asarray(signal, float), finalize_local_scan_statistics_np.asarray(background, float)
    width = float(kappa_log_width)
    invalid = dict(z_abcd=finalize_local_scan_statistics_np.nan, q0=finalize_local_scan_statistics_np.nan, fit_gradient=finalize_local_scan_statistics_np.nan, kappa_center=finalize_local_scan_statistics_np.nan,
                   kappa_fit=finalize_local_scan_statistics_np.nan, profile_delta=finalize_local_scan_statistics_np.nan, status='INVALID_INPUT')
    if (s.shape != (4,) or b.shape != (4,) or not finalize_local_scan_statistics_np.isfinite(s).all()
            or not finalize_local_scan_statistics_np.isfinite(b).all() or finalize_local_scan_statistics_np.any(s < 0) or finalize_local_scan_statistics_np.any(b <= 0)
            or finalize_local_scan_statistics_np.isnan(width) or width < 0):
        return invalid
    logk0 = float(finalize_local_scan_statistics_np.dot([1., -1., -1., 1.], finalize_local_scan_statistics_np.log(b)))
    kappa0 = float(finalize_local_scan_statistics_np.exp(logk0))
    n = b + s
    if not finalize_local_scan_statistics_np.isfinite(n).all() or finalize_local_scan_statistics_np.any(n <= 0):
        return invalid
    if finalize_local_scan_statistics_np.isinf(width):
        return dict(z_abcd=0., q0=0., fit_gradient=0., kappa_center=kappa0,
                    kappa_fit=float(n[0] * n[3] / (n[1] * n[2])),
                    profile_delta=0., status='OK_FREE_KAPPA')
    signs = finalize_local_scan_statistics_np.asarray([1., -1., -1., 1.])
    logn = finalize_local_scan_statistics_np.log(n)
    def equation(delta):
        return float(finalize_local_scan_statistics_np.dot(signs, finalize_local_scan_statistics_np.log(n + signs * delta)) - logk0 + width**2 * delta)
    if abs(equation(0.)) <= 2e-14:
        delta = 0.
    else:
        low = finalize_local_scan_statistics_np.nextafter(-min(n[0], n[3]), finalize_local_scan_statistics_np.inf)
        high = finalize_local_scan_statistics_np.nextafter(min(n[1], n[2]), -finalize_local_scan_statistics_np.inf)
        try:
            delta = finalize_local_scan_statistics_brentq(equation, low, high, xtol=1e-14, rtol=2e-14, maxiter=200)
        except (ValueError, RuntimeError, FloatingPointError):
            return dict(invalid, kappa_center=kappa0, status='FIT_FAILED')
    fitted = n + signs * delta
    log_ratio = finalize_local_scan_statistics_np.log(fitted) - logn
    deviance_half = float(finalize_local_scan_statistics_np.sum(n * (finalize_local_scan_statistics_np.expm1(log_ratio) - log_ratio)))
    logk_fit = float(finalize_local_scan_statistics_np.dot(signs, finalize_local_scan_statistics_np.log(fitted)))
    constraint = .5 * ((logk_fit - logk0) / width)**2 if width > 0 else 0.
    q0 = max(0., 2 * (deviance_half + constraint))
    if delta == 0. and abs(equation(0.)) <= 2e-14:
        q0 = 0.
    gradient = abs(equation(delta))
    if not finalize_local_scan_statistics_np.isfinite(q0) or gradient > 1e-6:
        return dict(invalid, kappa_center=kappa0, status='FIT_FAILED')
    return dict(z_abcd=float(finalize_local_scan_statistics_np.sqrt(q0)), q0=q0, fit_gradient=gradient,
                kappa_center=kappa0, kappa_fit=float(finalize_local_scan_statistics_np.exp(logk_fit)),
                profile_delta=float(delta), status='OK')


def finalize_local_scan_statistics_local_likelihood_scenarios(result, config):
    """Evaluate only explicitly enabled and configured MC-Asimov scenarios."""
    inputs = finalize_local_scan_statistics_local_likelihood_inputs(result, config)
    if not bool(config.get('likelihood_enabled', False)):
        return dict(likelihood_signals=finalize_local_scan_statistics_pd.DataFrame(), likelihood_summary=finalize_local_scan_statistics_pd.DataFrame(),
                    likelihood_inputs=inputs, likelihood_status=finalize_local_scan_statistics_pd.DataFrame([dict(
                        status='NOT_RUN', reason='No kappa-transfer assumption or uncertainty scenario has been adopted.',
                        input_rows=len(inputs), nominal_data_used=False)]))
    if 'kappa_log_widths' not in config or not len(config['kappa_log_widths']):
        raise ValueError('Enabled likelihood scenarios require explicit kappa_log_widths.')
    widths = tuple(float(value) for value in config['kappa_log_widths'])
    if len(widths) != len(set(widths)) or any(finalize_local_scan_statistics_np.isnan(width) or width < 0 for width in widths):
        raise ValueError('Each log-kappa width must be unique and nonnegative.')
    tables = result['tables']
    coordinates = tables['working_points'].sort_values('wp').reset_index(drop=True)
    signal = tables['signal_grid']
    signal = signal.loc[signal['sample'].str.startswith('2Mu2E_')]
    samples = sorted(signal['sample'].unique())
    if len(samples) != 60:
        raise ValueError('Likelihood scenarios require 60 matching-decay hypotheses.')
    background = tables['background_grid']
    rows = []
    for point in coordinates.itertuples():
        bg = background.loc[finalize_local_scan_statistics_np.isclose(background.tx, point.tx) & finalize_local_scan_statistics_np.isclose(background.ty, point.ty)]
        if len(bg) != 1:
            raise ValueError('Exactly one nominal background row is required per WP.')
        b = bg.iloc[0][list('ABCD')].to_numpy(float)
        chosen = signal.loc[finalize_local_scan_statistics_np.isclose(signal.tx, point.tx) & finalize_local_scan_statistics_np.isclose(signal.ty, point.ty)]
        for entry in chosen.to_dict('records'):
            s = finalize_local_scan_statistics_np.asarray([entry['S_' + region] for region in 'ABCD'], float)
            for width in widths:
                fit = finalize_local_scan_statistics_asimov_abcd_significance(s, b, width)
                rows.append(dict(wp=point.wp, label=point.label, tx=point.tx, ty=point.ty,
                                 is_reference=bool(point.is_reference), sample=entry['sample'],
                                 kappa_log_width=width, kappa_up_factor=float(finalize_local_scan_statistics_np.exp(width)), **fit))
    frame = finalize_local_scan_statistics_pd.DataFrame(rows)
    reference = frame.loc[frame.is_reference, ['sample', 'kappa_log_width', 'z_abcd', 'status']].rename(
        columns={'z_abcd': 'z_abcd_reference', 'status': 'reference_status'})
    frame = frame.merge(reference, on=['sample', 'kappa_log_width'], validate='many_to_one')
    frame['z_abcd_ratio'] = finalize_local_scan_statistics_np.nan
    summaries = []
    for width in widths:
        subset = frame.loc[frame.kappa_log_width.eq(width)]
        good = subset.status.isin(['OK', 'OK_FREE_KAPPA']) & subset.z_abcd_reference.gt(0)
        complete = subset.loc[good].groupby('sample').size()
        cohort = complete.index[complete.eq(len(coordinates))]
        eligible = frame.kappa_log_width.eq(width) & frame['sample'].isin(cohort)
        frame.loc[eligible, 'z_abcd_ratio'] = (frame.loc[eligible, 'z_abcd']
                                                   / frame.loc[eligible, 'z_abcd_reference'])
        for point in coordinates.itertuples():
            group = frame.loc[frame.kappa_log_width.eq(width) & frame.wp.eq(point.wp)]
            ratios = group.z_abcd_ratio.dropna()
            summaries.append(dict(wp=point.wp, label=point.label, tx=point.tx, ty=point.ty,
                                  kappa_log_width=width, kappa_up_factor=float(finalize_local_scan_statistics_np.exp(width)),
                                  hypotheses=len(group), z_abcd_median=group.z_abcd.median(),
                                  z_abcd_ratio_median=ratios.median(), z_abcd_ratio_q10=ratios.quantile(.1),
                                  z_abcd_ratio_q90=ratios.quantile(.9), valid_ratio_hypotheses=len(ratios),
                                  failed_fits=int((~group.status.isin(['OK', 'OK_FREE_KAPPA'])).sum())))
    return dict(likelihood_signals=frame, likelihood_summary=finalize_local_scan_statistics_pd.DataFrame(summaries),
                likelihood_inputs=inputs, likelihood_status=finalize_local_scan_statistics_pd.DataFrame([dict(
                    status='CONDITIONAL_MC_ASIMOV_SCENARIOS', nominal_data_used=False,
                    reason='Truth kappa is derived from nominal MC. Declared log-kappa widths are scenarios, not VR transfer calibrations. Sideband Poisson statistics and signal leakage are included; MC statistical nuisances are not included in this likelihood.')]))


def finalize_local_scan_statistics_validate_stats():
    """Check partition/covariance identities and the likelihood profile limits."""
    coordinates = finalize_local_scan_statistics_pd.DataFrame([(x, y) for x in (.2, .3) for y in (2.4, 2.8)], columns=['tx', 'ty'])
    xs, ys, _, masks = finalize_local_scan_statistics__local_partition(coordinates)
    atoms = finalize_local_scan_statistics_np.asarray([2., 3., 5., 7.])
    variance = finalize_local_scan_statistics_np.asarray([1., 4., 9., 16.])
    cumulative = masks @ atoms
    finalize_local_scan_statistics_np.testing.assert_allclose(finalize_local_scan_statistics__cumulative_to_atoms(cumulative, (len(xs), len(ys))), atoms)
    cov = (masks * variance) @ masks.T
    cumulative_v = masks @ variance
    for i, a in coordinates.iterrows():
        for j, b in coordinates.iterrows():
            intersection = finalize_local_scan_statistics_np.flatnonzero(finalize_local_scan_statistics_np.isclose(coordinates.tx, min(a.tx, b.tx))
                                           & finalize_local_scan_statistics_np.isclose(coordinates.ty, max(a.ty, b.ty)))
            finalize_local_scan_statistics_np.testing.assert_allclose(cov[i, j], cumulative_v[intersection[0]])
    draws, _ = finalize_local_scan_statistics__moment_draws(atoms, variance, 10, finalize_local_scan_statistics_np.random.default_rng(71))
    finalize_local_scan_statistics_np.testing.assert_array_equal(draws @ masks[[0, 0]].T[:, 0], draws @ masks[[0, 0]].T[:, 1])
    assert finalize_local_scan_statistics_np.isnan(finalize_local_scan_statistics_asimov_significance(1., 0.))
    b = finalize_local_scan_statistics_np.asarray([10., 20., 30., 60.])
    assert finalize_local_scan_statistics_asimov_abcd_significance(finalize_local_scan_statistics_np.zeros(4), b, .2)['z_abcd'] == 0
    assert finalize_local_scan_statistics_asimov_abcd_significance(b * .3, b, .2)['z_abcd'] == 0
    signal = finalize_local_scan_statistics_np.asarray([8., .5, .3, .1])
    answers = [finalize_local_scan_statistics_asimov_abcd_significance(signal, b, width) for width in (0., .1, .3, 1., finalize_local_scan_statistics_np.inf)]
    assert all(answer['status'] in ('OK', 'OK_FREE_KAPPA') for answer in answers)
    assert finalize_local_scan_statistics_np.all(finalize_local_scan_statistics_np.diff([answer['z_abcd'] for answer in answers]) <= 1e-10)
    assert answers[-1]['z_abcd'] == 0
    assert finalize_local_scan_statistics_asimov_abcd_significance(signal, [0, 20, 30, 60])['status'] == 'INVALID_INPUT'
    # Independent multidimensional convex fit checks the scalar score reduction.
    width = .3
    n = b + signal
    logk0 = finalize_local_scan_statistics_np.log(b[0] * b[3] / (b[1] * b[2]))
    def objective(parameters):
        beta, gamma, delta, logk = parameters
        loglam = finalize_local_scan_statistics_np.asarray([logk + beta + gamma - delta, beta, gamma, delta])
        difference = loglam - finalize_local_scan_statistics_np.log(n)
        cost = finalize_local_scan_statistics_np.sum(n * (finalize_local_scan_statistics_np.expm1(difference) - difference)) + .5 * ((logk - logk0) / width)**2
        residual = n * finalize_local_scan_statistics_np.expm1(difference)
        gradient = finalize_local_scan_statistics_np.asarray([residual[0] + residual[1], residual[0] + residual[2],
                               -residual[0] + residual[3], residual[0] + (logk - logk0) / width**2])
        return float(cost), gradient
    fitted = finalize_local_scan_statistics_minimize(objective, finalize_local_scan_statistics_np.r_[finalize_local_scan_statistics_np.log(b[1:]), logk0], jac=True, method='BFGS',
                      options={'gtol': 1e-9, 'maxiter': 1000})
    finalize_local_scan_statistics_np.testing.assert_allclose(2 * fitted.fun, answers[2]['q0'], rtol=1e-8, atol=1e-10)
    return dict(partition_reconstruction=True, intersection_covariance=True,
                shared_wp_draws=True, zero_background_undefined=True,
                zero_signal=True, proportional_signal_no_identifiability=True,
                unconstrained_kappa_no_identifiability=True,
                wider_kappa_constraint_reduces_z=True, invalid_inputs_rejected=True,
                independent_profile_fit_agrees=True)


def finalize_local_scan_statistics_add_local_statistics(result, config):
    """Append statistical diagnostics without changing primary MC WP selection."""
    before = result['tables']['working_points'].copy(deep=True)
    result['tables'].update(finalize_local_scan_statistics_local_rank_diagnostics(result, config))
    result['tables'].update(finalize_local_scan_statistics_local_likelihood_scenarios(result, config))
    validation = finalize_local_scan_statistics_pd.DataFrame([dict(check='Statistics: ' + check, passed=passed)
                               for check, passed in finalize_local_scan_statistics_validate_stats().items()])
    existing = result['tables'].get('validation', finalize_local_scan_statistics_pd.DataFrame())
    if 'check' in existing:
        existing = existing.loc[~existing['check'].fillna('').str.startswith('Statistics: ')]
    result['tables']['validation'] = finalize_local_scan_statistics_pd.concat([existing, validation], ignore_index=True)
    result.setdefault('metadata', {})['statistical_diagnostics'] = dict(
        n_toys=int(config.get('n_toys', 2000)), rng_seed=int(config.get('rng_seed', 73021)),
        resampling='Positive disjoint-bin scaled-Poisson moment approximation; not a raw-event bootstrap.',
        ranking_frequencies='Conditional on positive support at all 16 WPs and all 60 reference signal significances; not confidence probabilities.',
        covariance='Poisson independent-event sumw2 in the intersection of overlapping A selections.',
        likelihood_enabled=bool(config.get('likelihood_enabled', False)),
        likelihood_kappa_center='Nominal MC background odds ratio, conditional Asimov truth only.',
        likelihood_kappa_log_widths=list(config.get('kappa_log_widths', [])),
        likelihood_mc_statistical_constraints=False, vr_to_sr_kappa_transfer_adopted=False,
        nominal_data_used=False)
    finalize_local_scan_statistics_pd.testing.assert_frame_equal(before, result['tables']['working_points'])
    return result


def finalize_local_scan_statistics_local_likelihood_inputs(result, config):
    """Prepare MC-only fit inputs even when no kappa scenario has been chosen.

    Kappa and its propagated MC error are diagnostics, not an adopted nuisance
    constraint. Preparing this table performs no likelihood fit or DATA read.
    """
    tables = result['tables']
    coordinates = tables['working_points'].sort_values('wp').reset_index(drop=True)
    signal = tables['signal_grid']
    signal = signal.loc[signal['sample'].str.startswith('2Mu2E_')]
    if signal['sample'].nunique() != 60:
        raise ValueError('Likelihood inputs require 60 matching-decay hypotheses.')
    background = tables['background_grid']
    rows = []
    for point in coordinates.itertuples():
        selected = background.loc[finalize_local_scan_statistics_np.isclose(background.tx, point.tx)
                                  & finalize_local_scan_statistics_np.isclose(background.ty, point.ty)]
        if len(selected) != 1:
            raise ValueError('Exactly one nominal background input is required per WP.')
        bg = selected.iloc[0]
        b = bg[list('ABCD')].to_numpy(float)
        chosen = signal.loc[finalize_local_scan_statistics_np.isclose(signal.tx, point.tx) & finalize_local_scan_statistics_np.isclose(signal.ty, point.ty)]
        if len(chosen) != 60 or chosen['sample'].duplicated().any():
            raise ValueError('Each WP requires exactly 60 unique signal input rows.')
        for entry in chosen.to_dict('records'):
            s = finalize_local_scan_statistics_np.asarray([entry['S_' + region] for region in 'ABCD'], float)
            bg_valid = bool(finalize_local_scan_statistics_np.isfinite(b).all() and (b > 0).all())
            signal_valid = bool(finalize_local_scan_statistics_np.isfinite(s).all() and (s >= 0).all())
            rows.append(dict(
                wp=point.wp, label=point.label, tx=point.tx, ty=point.ty,
                sample=entry['sample'],
                **{'background_' + region: b[i] for i, region in enumerate('ABCD')},
                **{'background_var_' + region: float(bg.get('var_' + region, finalize_local_scan_statistics_np.nan)) for region in 'ABCD'},
                **{'signal_' + region: s[i] for i, region in enumerate('ABCD')},
                **{'signal_var_' + region: float(entry.get('var_S_' + region, finalize_local_scan_statistics_np.nan)) for region in 'ABCD'},
                kappa_mc=float(bg.get('kappa', finalize_local_scan_statistics_np.nan)),
                kappa_mc_error=float(bg.get('kappa_error', finalize_local_scan_statistics_np.nan)),
                min_sideband_neff=float(bg.get('min_sideband_neff', finalize_local_scan_statistics_np.nan)),
                min_abcd_neff=float(bg.get('neff_min', finalize_local_scan_statistics_np.nan)),
                background_neff_A=float(bg.get('neff_A', finalize_local_scan_statistics_np.nan)),
                low_mc_background_stat=bool(not finalize_local_scan_statistics_np.isfinite(bg.get('neff_min', finalize_local_scan_statistics_np.nan))
                                            or bg.get('neff_min', finalize_local_scan_statistics_np.nan) < config.get('min_neff', 10.)),
                background_positive=bg_valid, signal_nonnegative=signal_valid,
                input_valid=bg_valid and signal_valid,
                kappa_constraint_adopted=False, nominal_data_used=False,
                status='MC_INPUTS_ONLY_UNCALIBRATED_KAPPA' if bg_valid and signal_valid else 'INVALID_MC_INPUT'))
    frame = finalize_local_scan_statistics_pd.DataFrame(rows)
    if len(frame) != len(coordinates) * 60 or frame.duplicated(['wp', 'sample']).any():
        raise AssertionError('Likelihood input table must contain one row per WP and signal.')
    return frame


# ========================================================================
# abcd_mass_inclusive: original study API
# ========================================================================
"""Mass-inclusive 2mu2e validation with explicit PP/PD/DP DATA safeguards.

The physical mass range starts at zero and has no upper cut. Numeric mass
underflow is excluded, and mass overflow belongs to the >=300 GeV component.
Nominal DD is projected for MC only; no nominal DATA projection is available.
"""
from pathlib import Path as finalize_mass_inclusive_Path
import gc as finalize_mass_inclusive_gc

import numpy as finalize_mass_inclusive_np
import pandas as finalize_mass_inclusive_pd

finalize_mass_inclusive_common = _FINALIZE_NAMESPACES['abcd_finalize_common']


finalize_mass_inclusive_STATES = ('PP', 'PD', 'DP')
finalize_mass_inclusive_MASS_SLICES = (('low', 0., 150.), ('intermediate', 150., 300.), ('high', 300., None))


def finalize_mass_inclusive__settings(config):
    if config['topology'] != '2mu2e' or config['purity_limit'] != .1:
        raise ValueError('Mass-inclusive validation requires 2mu2e and S/B < 0.10.')
    if (config.get('partner_iso_low', 0.), config.get('partner_iso_high', .5)) != (0., .5):
        raise ValueError('Keep partner isolation in [0, 0.5).')
    background = finalize_mass_inclusive_Path(config['bkg_dir'])
    if '_bkg_' not in background.name:
        raise ValueError('Expected the established background production directory.')
    return dict(config, bkg_dir=background,
                signal_dir=background.with_name(background.name.replace('_bkg_', '_signal_')),
                data_dir=background.with_name(background.name.replace('_bkg_', '_data_')),
                candidate_order=list(finalize_mass_inclusive_STATES), mass_min=0., mass_max=None,
                vr='MassInclusive', tx=.3, ty=2.6)


def finalize_mass_inclusive__specs(config):
    return {state: dict(
        state0=state[0], state1=state[1], partner_iso_low=0., partner_iso_high=.5,
        mass_low=0., mass_high=None, tx=.3, ty=2.6,
        scan_mu_iso=list(config['scan_mu_iso']), scan_abs_dphi=list(config['scan_abs_dphi']),
        plateau_mu_iso=[], plateau_abs_dphi=[], kind='mu_dphi',
        label=f'All physical MJJ {state}: {state[0]} / {state[1]}',
        selection='0 <= partner iso < 0.5; all physical MJJ >= 0, no upper cut',
        x_label='Mu-LJ isolation', y_label='|Δφ| [rad]') for state in finalize_mass_inclusive_STATES}


def finalize_mass_inclusive__state_index(edges, state, config):
    """Whitelist before accessing weights; numeric displacement flows are kept."""
    if config['topology'] != '2mu2e' or state not in finalize_mass_inclusive_STATES:
        raise ValueError('Only 2mu2e PP, PD and DP are allowed; DD/inclusive states are forbidden.')
    partner = slice(finalize_mass_inclusive_common._comparison_split(edges['iso1'], 0.),
                    finalize_mass_inclusive_common._comparison_split(edges['iso1'], .5))
    mu_split = finalize_mass_inclusive_common._comparison_split(edges['disp0'], 2.5)
    egm_split = finalize_mass_inclusive_common._comparison_split(edges['disp1'], .5)
    first = slice(mu_split, None) if state[0] == 'P' else slice(None, mu_split)
    second = slice(None, egm_split) if state[1] == 'P' else slice(egm_split, None)
    return (slice(None), partner, first, second, slice(None))


def finalize_mass_inclusive_project_mass_partitions(weights, variances, edges, state, config):
    """MC-only mass partition plus an excluded-underflow audit, retaining flows."""
    index = finalize_mass_inclusive__state_index(edges, state, config) + (slice(None),)
    selected = [array[index] for array in (weights, variances)]
    finalize_mass_inclusive_common._comparison_validate_arrays(*selected)
    # Keep only iso0, angle and mass, then partition mass on native boundaries.
    reduced = [array.sum(axis=(1, 2, 3)) for array in selected]
    partitions = {}
    for name, lower, upper in finalize_mass_inclusive_MASS_SLICES:
        low = finalize_mass_inclusive_common._comparison_split(edges['ljlj_mass'], lower)
        high = None if upper is None else finalize_mass_inclusive_common._comparison_split(edges['ljlj_mass'], upper)
        planes = [array[:, :, low:high].sum(axis=2) for array in reduced]
        partitions[name] = dict(finalize_mass_inclusive_common._comparison_plane(*planes, edges['iso0'], edges['abs_dphi']),
                                kind='mu_dphi')
    physical_low = finalize_mass_inclusive_common._comparison_split(edges['ljlj_mass'], 0.)
    all_arrays = [sum(partitions[name][field] for name, _, _ in finalize_mass_inclusive_MASS_SLICES)
                  for field in ('values', 'variances')]
    for actual, original in zip(all_arrays, reduced):
        finalize_mass_inclusive_np.testing.assert_allclose(actual, original[:, :, physical_low:].sum(axis=2),
                                   rtol=1e-10, atol=1e-8,
                                   err_msg='Mass slices do not partition the physical all-mass plane.')
    partitions['all'] = dict(finalize_mass_inclusive_common._comparison_plane(*all_arrays, edges['iso0'], edges['abs_dphi']),
                             kind='mu_dphi')
    # Match ABCD iso support in this MC-only flow audit. DATA never uses this path.
    audit = dict(mass_underflow_sumw=float(reduced[0][1:, :, :physical_low].sum()),
                 mass_underflow_sumw2=float(reduced[1][1:, :, :physical_low].sum()),
                 physical_mass_sumw=float(all_arrays[0][1:, :].sum()),
                 physical_mass_sumw2=float(all_arrays[1][1:, :].sum()))
    return partitions, audit


def finalize_mass_inclusive__project_all_data(weights, variances, edges, state, config):
    """Project only the authorized all-mass P-containing VR, never mass sub-VRs."""
    low = finalize_mass_inclusive_common._comparison_split(edges['ljlj_mass'], 0.)
    index = finalize_mass_inclusive__state_index(edges, state, config) + (slice(low, None),)
    selected = [array[index] for array in (weights, variances)]
    finalize_mass_inclusive_common._comparison_validate_arrays(*selected)
    reduced = [array.sum(axis=(1, 2, 3, 5)) for array in selected]
    for original, plane in zip(selected, reduced):
        finalize_mass_inclusive_np.testing.assert_allclose(plane.sum(), original.sum(), rtol=1e-10, atol=1e-8)
    return dict(finalize_mass_inclusive_common._comparison_plane(*reduced, edges['iso0'], edges['abs_dphi']), kind='mu_dphi')


def finalize_mass_inclusive__load_mc(config):
    backgrounds = sorted(config['bkg_dir'].glob('*.coffea'))
    if len(backgrounds) != 18:
        raise ValueError('Expected the same 18 background MC files.')
    signals = finalize_mass_inclusive_common.candidate_signal_paths(config)
    mass_backgrounds = {state: {name: None for name in ('low', 'intermediate', 'high', 'all')}
                        for state in finalize_mass_inclusive_STATES}
    mass_signals = {state: {name: {} for name in ('low', 'intermediate', 'high', 'all')}
                   for state in finalize_mass_inclusive_STATES}
    sr_plane, sr_samples, sr_signals = None, {}, {}
    manifest, flows, validations = [], [], []
    jobs = [(path, 'Background') for path in backgrounds] + [(path, 'Signal hypothesis') for path in signals]
    for index, (path, source) in enumerate(jobs, 1):
        loaded, payload = finalize_mass_inclusive_common._comparison_payload(path, 'sixd')
        weights, variances, edges = finalize_mass_inclusive_common._comparison_current_cube(payload, path, config)
        for state in finalize_mass_inclusive_STATES:
            partitions, audit = finalize_mass_inclusive_project_mass_partitions(weights, variances, edges, state, config)
            flows.append(dict(sample=path.stem, source=source, plane=state, **audit))
            for name, plane in partitions.items():
                if source == 'Background':
                    mass_backgrounds[state][name] = finalize_mass_inclusive_common.add_candidate(
                        mass_backgrounds[state][name], plane, finalize_mass_inclusive_common._comparison_group(path.stem))
                else:
                    mass_signals[state][name][path.stem] = plane
            validations.append(dict(scope=f'MassInclusive_{state}', sample=path.stem,
                                    check='Three mass slices partition all physical mass sumw and sumw2', passed=True))
        nominal = finalize_mass_inclusive_common.project_mc_sr(weights, variances, edges, config)
        if source == 'Background':
            sr_plane = finalize_mass_inclusive_common.add_candidate(sr_plane, nominal, finalize_mass_inclusive_common._comparison_group(path.stem))
            sr_samples[path.stem] = nominal
        else:
            sr_signals[path.stem] = nominal
        manifest.append(finalize_mass_inclusive_common.candidate_input_record(path, source))
        del loaded, payload, weights, variances
        finalize_mass_inclusive_gc.collect()
        if index % 20 == 0 or index == len(jobs):
            print(f'Mass-inclusive MC {index}/{len(jobs)}', flush=True)
    return dict(backgrounds=mass_backgrounds, signals=mass_signals, sr_plane=sr_plane,
                sr_sample_planes=sr_samples, sr_signals=sr_signals,
                manifest=manifest, mass_flow_audit=finalize_mass_inclusive_pd.DataFrame(flows), validation=validations)


def finalize_mass_inclusive__validate_middle_mass(mc, config, points):
    """Regress raw intermediate-mass MC against the existing notebook-03 cache."""
    directory = finalize_mass_inclusive_Path(config['vr2_output_dir'])
    grid_path, signal_path = directory / 'grid.csv', directory / 'signal_points.csv'
    grid = finalize_mass_inclusive_pd.read_csv(grid_path, float_precision='round_trip')
    grid = grid.loc[grid.source.eq('MC')]
    signals = finalize_mass_inclusive_pd.read_csv(signal_path, float_precision='round_trip')
    selected = points[['tx', 'ty']].drop_duplicates()
    signals = selected.merge(signals, on=['tx', 'ty'], validate='one_to_many')
    if len(signals) != len(selected) * 3 * 120:
        raise AssertionError('The cached intermediate-mass signals do not cover every selected point.')
    validation = []
    fields = list('ABCD') + ['var_' + region for region in 'ABCD']
    for state in finalize_mass_inclusive_STATES:
        cached = grid.loc[grid.plane.eq(state)]
        if len(cached) != 154:
            raise AssertionError('Intermediate-mass MC regression requires all 154 native points.')
        plane = mc['backgrounds'][state]['intermediate']
        for row in cached.itertuples():
            actual = finalize_mass_inclusive_common.plane_statistics(plane, row.tx, row.ty)
            finalize_mass_inclusive_np.testing.assert_allclose([actual[field] for field in fields],
                                       [getattr(row, field) for field in fields],
                                       rtol=1e-8, atol=1e-8, equal_nan=True)
        for row in signals.loc[signals.plane.eq(state)].itertuples():
            regions = finalize_mass_inclusive_common._comparison_regions(mc['signals'][state]['intermediate'][row.sample], row.tx, row.ty)
            for region in 'ABCD':
                finalize_mass_inclusive_np.testing.assert_allclose([regions[region]['value'], regions[region]['variance']],
                                           [getattr(row, 'S_' + region), getattr(row, 'var_S_' + region)],
                                           rtol=1e-8, atol=1e-8)
        validation.append(dict(scope=f'VR2_{state}',
                               check='Intermediate mass raw MC: 154 background WPs and 120 signals at each selected WP reproduce notebook 03',
                               passed=True))
    manifest = [finalize_mass_inclusive_common.candidate_input_record(grid_path, 'Intermediate-mass MC regression cache'),
                finalize_mass_inclusive_common.candidate_input_record(signal_path, 'Intermediate-mass signal regression cache')]
    return validation, manifest


def finalize_mass_inclusive__mass_component_table(mc, points):
    rows = []
    for point in points.itertuples():
        for state in finalize_mass_inclusive_STATES:
            full = finalize_mass_inclusive_common._comparison_regions(mc['backgrounds'][state]['all'], point.tx, point.ty)
            for name, low, high in finalize_mass_inclusive_MASS_SLICES:
                background = finalize_mass_inclusive_common._comparison_regions(mc['backgrounds'][state][name], point.tx, point.ty)
                signal_regions = {sample: finalize_mass_inclusive_common._comparison_regions(plane, point.tx, point.ty)
                                  for sample, plane in mc['signals'][state][name].items()}
                if len(signal_regions) != 120:
                    raise AssertionError('Mass-component contamination requires all 120 hypotheses.')
                for region in 'ABCD':
                    b, vb = background[region]['value'], background[region]['variance']
                    valid_b = bool(finalize_mass_inclusive_np.isfinite(b) and b > 1e-8)
                    candidates = [(sample, item[region]['value'], item[region]['variance'])
                                  for sample, item in signal_regions.items()]
                    valid_signals = all(finalize_mass_inclusive_np.isfinite(s) and s >= 0 for _, s, _ in candidates)
                    sample, s, vs = max(candidates, key=lambda item: item[1])
                    ratio = s / b if valid_b and valid_signals else finalize_mass_inclusive_np.nan
                    error = finalize_mass_inclusive_np.sqrt(vs / b**2 + s*s*vb / b**4) if finalize_mass_inclusive_np.isfinite(ratio) else finalize_mass_inclusive_np.nan
                    total = full[region]['value']
                    rows.append(dict(wp=point.wp, tx=point.tx, ty=point.ty, plane=state,
                                     mass_slice=name, mass_low=low, mass_high=high, region=region,
                                     background=b, var_background=vb, background_error=finalize_mass_inclusive_np.sqrt(vb),
                                     neff=b*b/vb if vb > 0 else finalize_mass_inclusive_np.nan,
                                     background_fraction=b/total if total > 0 else finalize_mass_inclusive_np.nan,
                                     max_S_B=ratio, max_S_B_error=error,
                                     worst_sample=sample if finalize_mass_inclusive_np.isfinite(ratio) else '', signal_max=s,
                                     signal_hypotheses=120, valid_background=valid_b,
                                     valid_signals=bool(valid_signals),
                                     approved=bool(valid_b and valid_signals and ratio < .1)))
    return finalize_mass_inclusive_pd.DataFrame(rows)


def finalize_mass_inclusive__load_data(config, gates):
    # This list is built only after integrated and displayed-bin MC checks.
    allowed = [state for state in finalize_mass_inclusive_STATES
               if any(gate['approved'] for (name, _, _), gate in gates.items() if name == state)]
    planes, manifest = {state: None for state in finalize_mass_inclusive_STATES}, []
    if not allowed:
        print('No mass-inclusive WP passes the MC gate; no DATA files opened.', flush=True)
        return planes, manifest
    paths = sorted(config['data_dir'].glob('*.coffea'))
    if len(paths) != 68:
        raise ValueError('Expected the same 68 DATA files.')
    for index, path in enumerate(paths, 1):
        loaded, payload = finalize_mass_inclusive_common._extension_data_payload(path)
        weights, variances, edges = finalize_mass_inclusive_common._comparison_current_cube(payload, path, config)
        for state in allowed:
            # No nominal projector, DD selector, or mass-component DATA sums here.
            plane = finalize_mass_inclusive__project_all_data(weights, variances, edges, state, config)
            planes[state] = finalize_mass_inclusive_common.add_candidate(planes[state], plane, 'Total')
        manifest.append(finalize_mass_inclusive_common.candidate_input_record(path, 'DATA mass-inclusive PP/PD/DP only'))
        del loaded, payload, weights, variances
        finalize_mass_inclusive_gc.collect()
        if index % 20 == 0 or index == len(paths):
            print(f'Mass-inclusive approved VR DATA {index}/{len(paths)}', flush=True)
    return planes, manifest


def finalize_mass_inclusive__validate_tables(tables, config, points):
    validation = []
    fields = list('ABCD') + ['var_' + region for region in 'ABCD'] + ['pred', 'pred_error', 'kappa', 'kappa_error']
    for state in finalize_mass_inclusive_STATES:
        data = tables['grid'].loc[tables['grid'].plane.eq(state) & tables['grid'].source.eq('DATA')]
        if len(data) != 154 or data.duplicated(['tx', 'ty']).any():
            raise AssertionError('Expected 154 unique mass-inclusive DATA gate records.')
        if not data.loc[~data.approved, fields].isna().all().all():
            raise AssertionError('A blocked mass-inclusive DATA value is exposed.')
        bins = tables['distribution_bin_purity'].loc[tables['distribution_bin_purity'].plane.eq(state)]
        expected = (bins.valid_background & bins.valid_signal_hypotheses.eq(120)
                    & bins.max_S_B.lt(config['purity_limit']))
        if not expected.equals(bins.approved):
            raise AssertionError('The displayed-bin DATA mask does not match the MC gate.')
        validation.append(dict(scope=f'MassInclusive_{state}',
                               check='Complete 154-WP DATA gating and exact displayed-bin 120-model mask', passed=True))
    components = tables['mass_components']
    expected_rows = len(points) * 3 * 3 * 4
    if len(components) != expected_rows:
        raise AssertionError('Missing selected-point mass-component diagnostics.')
    selected_mc = tables['selected_closure'].loc[tables['selected_closure'].source.eq('MC')]
    totals = components.groupby(['wp', 'plane', 'region'])[['background', 'var_background']].sum()
    for row in selected_mc.itertuples():
        for region in 'ABCD':
            actual = totals.loc[(row.wp, row.plane, region)]
            finalize_mass_inclusive_np.testing.assert_allclose(actual.to_numpy(float),
                                       [getattr(row, region), getattr(row, 'var_' + region)],
                                       rtol=1e-10, atol=1e-8)
    validation.append(dict(scope='MassInclusive',
                           check='Selected ABCD sumw/sumw2 equals the sum of low/intermediate/high mass components', passed=True))
    validation.append(dict(scope='DATA', check='Only explicit PP/PD/DP all-mass projections; no DD or nominal DATA projector', passed=True))
    return validation


def finalize_mass_inclusive_run_mass_inclusive(config, points):
    """Read MC once, calculate every gate, then load only disjoint approved VR DATA."""
    config = finalize_mass_inclusive__settings(config)
    points = points[['wp', 'tx', 'ty']].copy()
    if points.wp.duplicated().any() or points.duplicated(['tx', 'ty']).any():
        raise ValueError('Working-point labels and coordinates must be unique.')
    if not ((points.tx == .3) & (points.ty == 2.6)).any():
        raise ValueError('Include the unchanged (0.30, 2.6) reference.')
    specs = finalize_mass_inclusive__specs(config)
    mc = finalize_mass_inclusive__load_mc(config)
    validation, cache_manifest = finalize_mass_inclusive__validate_middle_mass(mc, config, points)
    planes = {state: {'VR': mc['backgrounds'][state]['all']} for state in finalize_mass_inclusive_STATES}
    signals = {state: mc['signals'][state]['all'] for state in finalize_mass_inclusive_STATES}
    gates, signal_points, region_purity = finalize_mass_inclusive_common.candidate_purity(planes, signals, specs, config)
    distribution_purity, distribution_table = finalize_mass_inclusive_common.candidate_distribution_purity(planes, signals, specs, config)
    data_planes, data_manifest = finalize_mass_inclusive__load_data(config, gates)
    tables = finalize_mass_inclusive_common.collect_candidate_tables(planes, data_planes, gates, specs, config)
    tables.update(distribution_bin_purity=distribution_table,
                  purity_grid_regions=region_purity,
                  purity_regions=region_purity.loc[finalize_mass_inclusive_np.isclose(region_purity.tx, .3)
                                                  & finalize_mass_inclusive_np.isclose(region_purity.ty, 2.6)].copy(),
                  mass_components=finalize_mass_inclusive__mass_component_table(mc, points),
                  mass_flow_audit=mc['mass_flow_audit'],
                  selection=finalize_mass_inclusive_pd.DataFrame([dict(plane=state, **spec) for state, spec in specs.items()]))
    tables['selected_closure'] = points.merge(tables['grid'], on=['tx', 'ty'], validate='one_to_many')
    tables['selected_purity'] = points.merge(region_purity, on=['tx', 'ty'], validate='one_to_many')
    if len(tables['selected_closure']) != len(points)*3*2 or len(tables['selected_purity']) != len(points)*3*4:
        raise AssertionError('Selected WPs must be available in every explicit VR category.')
    validation = mc['validation'] + validation + finalize_mass_inclusive__validate_tables(tables, config, points)
    tables['validation'] = finalize_mass_inclusive_pd.DataFrame(validation)
    tables['input_manifest'] = finalize_mass_inclusive_pd.DataFrame(mc['manifest'] + cache_manifest + data_manifest + [
        finalize_mass_inclusive_common.candidate_input_record(finalize_mass_inclusive_Path(__file__), 'Mass-inclusive implementation')])
    metadata = dict(
        mass_selection='All physical MJJ >= 0 with no upper cut; numeric mass underflow excluded, overflow retained.',
        mass_components=['0<=MJJ<150', '150<=MJJ<300', 'MJJ>=300 including overflow'],
        states=list(finalize_mass_inclusive_STATES), partner_iso='0<=iso<0.5',
        nominal_data_loaded=False, nominal_data_projected=False,
        original_background_files_opened=18, original_signal_files_opened=120,
        original_data_files_opened=len(data_manifest),
        data_gate='All 120 separate signals x ABCD must have S/B_MC<0.10 and positive MC support at the WP.',
        distribution_gate='Same exact display folding as the original plots; per-bin max S/B<0.10 and valid MC support, AND the selected integrated gate.',
        data_plane_policy='Internal all-mass PP/PD/DP planes may serve multiple approved WPs; apply each selected-WP gate before plotting. Do not export raw DATA arrays.',
        mass_component_policy='Mass components and underflow audit are MC only. No mass-component DATA sums are calculated.',
        mass_signal_policy='A component-wise maximum may belong to a different signal hypothesis in each mass slice and must not be added. All-mass S/B is computed after summing the three planes for each individual hypothesis.',
        normalization='Stored production weights, unchanged; distinct signal hypotheses are never summed.')
    return dict(config=config, specs=specs, selected=tables['selection'][['plane', 'tx', 'ty']].copy(),
                definitions=metadata, metadata=metadata, planes=planes, data_planes=data_planes,
                distribution_purity=distribution_purity, tables=tables,
                sr_plane=mc['sr_plane'], sr_sample_planes=mc['sr_sample_planes'], sr_signals=mc['sr_signals'],
                mass_background_planes=mc['backgrounds'], mass_signal_planes=mc['signals'],
                signal_points=signal_points)


# ========================================================================
# abcd_sideband_tradeoff: original study API
# ========================================================================
"""MC-only comparison of relaxed boundaries and WP2-A-preserving sideband gaps.

Every rectangular yield is read from the existing, validated nominal MC caches.
No original DATA file or nominal DATA yield is accessed. The 120 hypotheses are
kept separate, and the same-model sensitivity comparison uses 60 matching models.
"""
from pathlib import Path as finalize_sideband_tradeoff_Path
import json as finalize_sideband_tradeoff_json

import numpy as finalize_sideband_tradeoff_np
import pandas as finalize_sideband_tradeoff_pd

finalize_sideband_tradeoff_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_sideband_tradeoff_review = _FINALIZE_NAMESPACES['review_mc_sr_working_points']


finalize_sideband_tradeoff_REGIONS = tuple('ABCD')
finalize_sideband_tradeoff_BASELINES = (('Reference', .3, 2.6), ('WP1', .2, 2.6), ('WP2', .2, 2.7))
finalize_sideband_tradeoff_PROTECTED = ((.3, 2.6), (.2, 2.6), (.2, 2.7), (.2, 2.8), (.2, 2.4), (.3, 2.7))


def finalize_sideband_tradeoff_run_sideband_tradeoff(config):
    """Return nominal MC scans and an MC-only diagnostic shortlist.

    Loose candidates expand A relative to WP2. Gap candidates keep WP2 A
    exactly unchanged and move only the common high-iso / low-angle sidebands.
    Passing purity and geometric protection is necessary, not DATA permission
    or final-inference validation. No closure-based DATA ranking is performed.
    """
    if config['topology'] != '2mu2e' or config['purity_limit'] != .1:
        raise ValueError('This scan is restricted to 2mu2e with the existing strict 10% gate.')
    if (config['partner_iso_low'], config['partner_iso_high']) != (0., .5):
        raise ValueError('Partner isolation must remain [0,0.5).')
    folder = finalize_sideband_tradeoff_Path(config['vr2_output_dir'])
    manifest, validation = [], []
    path = folder / 'manifest.json'
    document = finalize_sideband_tradeoff_json.loads(path.read_text())
    settings = document['settings']
    if (settings['topology'], settings['current_hist'], settings['expected_current_channel']) != (
            '2mu2e', config['current_hist'], config['expected_current_channel']):
        raise ValueError('The nominal cache histogram or production selection changed.')
    if (settings['partner_iso_low'], settings['partner_iso_high'], settings['purity_limit']) != (0., .5, .1):
        raise ValueError('Cached nominal conventions differ from this scan.')
    manifest.append(finalize_sideband_tradeoff_common.candidate_input_record(path, 'Sideband nominal cache conventions'))
    background = finalize_sideband_tradeoff__read(folder / 'sr_reference_grid.csv', manifest)
    signal = finalize_sideband_tradeoff__read(folder / 'sr_reference_signal_points.csv', manifest)
    if len(background) != 154 or background.duplicated(['tx', 'ty']).any():
        raise AssertionError('Require the existing 154-point nominal MC grid.')
    if not background.source.eq('MC').all() or not background.context.eq('SR').all():
        raise AssertionError('Only nominal background MC rows may be loaded.')
    if len(signal) != 154*120 or signal['sample'].nunique() != 120 or signal.duplicated(['tx', 'ty', 'sample']).any():
        raise AssertionError('Require 120 separate signal hypotheses at every nominal grid point.')
    if not signal.context.eq('SR').all():
        raise AssertionError('Only nominal signal MC rows may be loaded.')
    values = signal[['S_'+r for r in finalize_sideband_tradeoff_REGIONS] + ['var_S_'+r for r in finalize_sideband_tradeoff_REGIONS]].to_numpy(float)
    if not finalize_sideband_tradeoff_np.isfinite(values).all() or (values < 0).any():
        raise AssertionError('Signal yields and sumw2 must be finite and nonnegative.')
    all_significance = finalize_sideband_tradeoff_review.signal_significance('2mu2e', background, signal)
    blookup = {(float(row.tx), float(row.ty)): row for row in background.itertuples()}
    slookup = {(float(tx), float(ty)): group.set_index('sample').sort_index()
               for (tx, ty), group in signal.groupby(['tx', 'ty'])}
    wp2_signal = slookup[(.2, 2.7)]
    wp2_background = blookup[(.2, 2.7)]
    ref_background = blookup[(.3, 2.6)]
    ref_signal = slookup[(.3, 2.6)]
    identifiers = wp2_signal.index
    if not all(frame.index.equals(identifiers) for frame in slookup.values()):
        raise AssertionError('Every coordinate must contain the identical signal hypotheses.')
    reference_z = all_significance.loc[all_significance.tx.eq(.2) & all_significance.ty.eq(2.7)].set_index('sample').reindex(identifiers)
    specifications = finalize_sideband_tradeoff_candidate_specifications(background)
    scan, regions, signals, geometry = [], [], [], []
    for spec in specifications:
        coordinates = finalize_sideband_tradeoff_candidate_coordinates(spec)
        yields = {r: float(getattr(blookup[coordinates[r]], r)) for r in finalize_sideband_tradeoff_REGIONS}
        variances = {r: float(getattr(blookup[coordinates[r]], 'var_'+r)) for r in finalize_sideband_tradeoff_REGIONS}
        statistics = finalize_sideband_tradeoff_statistics_from_regions(yields, variances, config.get('min_neff', 10.))
        source_signals = {r: slookup[coordinates[r]][['S_'+r, 'var_S_'+r]] for r in finalize_sideband_tradeoff_REGIONS}
        selected = finalize_sideband_tradeoff_pd.concat([source_signals[r] for r in finalize_sideband_tradeoff_REGIONS], axis=1).reindex(identifiers)
        selected.insert(0, 'sample', selected.index)
        selected.reset_index(drop=True, inplace=True)
        selected['matching_decay'] = selected['sample'].str.startswith('2Mu2E_')
        selected['z_known_b'] = finalize_sideband_tradeoff_review.asimov_significance(selected.S_A, yields['A'])
        selected['z_mc_stat_diagnostic'] = finalize_sideband_tradeoff_review.asimov_significance(selected.S_A, yields['A'], variances['A'])
        selected['signal_retention_WP2'] = finalize_sideband_tradeoff_review.finite_ratio(selected.S_A, selected['sample'].map(wp2_signal.S_A))
        selected['signal_retention_reference'] = finalize_sideband_tradeoff_review.finite_ratio(selected.S_A, selected['sample'].map(ref_signal.S_A))
        for metric in ('z_known_b', 'z_mc_stat_diagnostic'):
            selected[metric+'_ratio_WP2'] = finalize_sideband_tradeoff_review.finite_ratio(selected[metric], selected['sample'].map(reference_z[metric]))
        region_rows = []
        for region in finalize_sideband_tradeoff_REGIONS:
            s = selected['S_'+region].to_numpy(float)
            vs = selected['var_S_'+region].to_numpy(float)
            b, vb = yields[region], variances[region]
            valid = bool(finalize_sideband_tradeoff_np.isfinite(b) and b > 1e-8 and finalize_sideband_tradeoff_np.isfinite(vb) and vb >= 0)
            ratios = s / b if valid else finalize_sideband_tradeoff_np.full(len(s), finalize_sideband_tradeoff_np.nan)
            selected['S_B_'+region] = ratios
            selected['purity_pass_'+region] = finalize_sideband_tradeoff_np.isfinite(ratios) & (ratios < .1)
            worst_index = int(finalize_sideband_tradeoff_np.argmax(s))
            maximum = float(ratios[worst_index]) if valid else finalize_sideband_tradeoff_np.nan
            maximum_error = float(finalize_sideband_tradeoff_np.sqrt(vs[worst_index]/b**2 + s[worst_index]**2*vb/b**4)) if valid else finalize_sideband_tradeoff_np.nan
            overlap = finalize_sideband_tradeoff_region_geometry(spec, region)
            item = dict(spec, region=region, background=b, var_background=vb,
                        background_error=finalize_sideband_tradeoff_np.sqrt(vb), neff_background=statistics['neff_'+region],
                        max_S_B=maximum, max_S_B_error=maximum_error,
                        worst_sample=selected.iloc[worst_index]['sample'],
                        signal_max=float(s[worst_index]), var_signal_max=float(vs[worst_index]),
                        valid_background=valid, signal_hypotheses=120,
                        purity_pass=bool(valid and finalize_sideband_tradeoff_np.isfinite(ratios).all() and (ratios < .1).all()),
                        overlap_original_A=overlap['overlap_original_A'],
                        overlap_protected_A=overlap['overlap_protected_A'],
                        nominal_data_evaluated=False)
            region_rows.append(item)
            geometry.append(dict(spec, region=region, **overlap))
        regions.extend(region_rows)
        selected['leakage_bias'] = ((1+selected.S_B_B)*(1+selected.S_B_C)/(1+selected.S_B_D)-1)
        selected['candidate_id'] = spec['candidate_id']
        selected['strategy'] = spec['strategy']
        signals.extend(selected.to_dict('records'))
        sidebands = [row for row in region_rows if row['region'] in 'BCD']
        valid_sidebands = all(row['valid_background'] for row in sidebands)
        purity_pass = all(row['purity_pass'] for row in sidebands)
        geometry_safe = not any(row['overlap_protected_A'] for row in sidebands)
        worst = max(sidebands, key=lambda row: row['max_S_B']) if valid_sidebands else None
        row = dict(spec, **statistics, source='MC', signal_hypotheses=120,
                   max_S_B_BCD=worst['max_S_B'] if worst else finalize_sideband_tradeoff_np.nan,
                   worst_BCD_region=worst['region'] if worst else '',
                   worst_BCD_sample=worst['worst_sample'] if worst else '',
                   purity_pass=purity_pass, geometry_safe=geometry_safe,
                   operational_pass=bool(purity_pass and geometry_safe),
                   positive_BCD=valid_sidebands, nominal_data_evaluated=False,
                   background_retention_WP2=yields['A']/wp2_background.A,
                   background_retention_reference=yields['A']/ref_background.A,
                   prediction_relative_error=(statistics['pred_error']/statistics['pred']
                                              if statistics['pred'] > 0 else finalize_sideband_tradeoff_np.nan),
                   sideband_effective_precision=(statistics['pred']**2/statistics['pred_error']**2
                                                if statistics['pred_error'] > 0 else finalize_sideband_tradeoff_np.nan),
                   leakage_bias_max=float(selected.leakage_bias.max()),
                   leakage_bias_min=float(selected.leakage_bias.min()),
                   purity_margin=.1-worst['max_S_B'] if worst else finalize_sideband_tradeoff_np.nan)
        for item in sidebands:
            row['max_S_B_'+item['region']] = item['max_S_B']
        matching = selected.loc[selected.matching_decay]
        if len(matching) != 60:
            raise AssertionError('The sensitivity summary must contain exactly 60 matching models.')
        for field in ('signal_retention_WP2', 'signal_retention_reference', 'z_known_b',
                      'z_mc_stat_diagnostic', 'z_known_b_ratio_WP2', 'z_mc_stat_diagnostic_ratio_WP2'):
            finalize_sideband_tradeoff__quantiles(row, field, matching[field])
        if spec['strategy'] == 'gap':
            for region_field in ('A', 'var_A', 'A_error', 'neff_A'):
                finalize_sideband_tradeoff_np.testing.assert_allclose(statistics[region_field], getattr(wp2_background, region_field), rtol=1e-12)
            finalize_sideband_tradeoff_np.testing.assert_allclose(selected.S_A, wp2_signal.S_A, rtol=1e-12)
            for metric in ('z_known_b', 'z_mc_stat_diagnostic'):
                finalize_sideband_tradeoff_np.testing.assert_allclose(selected[metric], reference_z[metric], rtol=1e-12, equal_nan=True)
            validation.append(dict(scope=spec['candidate_id'], check='Gap retains WP2 A background, all 120 signals and both A-based significance diagnostics exactly', passed=True))
        else:
            cached = blookup[(spec['a_iso'], spec['a_phi'])]
            for field in [*finalize_sideband_tradeoff_REGIONS, *['var_'+r for r in finalize_sideband_tradeoff_REGIONS], 'pred', 'pred_error', 'kappa', 'kappa_error']:
                finalize_sideband_tradeoff_np.testing.assert_allclose(statistics[field], getattr(cached, field), rtol=1e-9, atol=1e-9, equal_nan=True)
            validation.append(dict(scope=spec['candidate_id'], check='Standard-boundary regional yields, prediction and kappa reproduce the existing nominal MC cache', passed=True))
        row['gate_reason'] = ('PURITY_AND_GEOMETRY_PASS_MC_ONLY' if row['operational_pass'] else
                              'NONPOSITIVE_MC_SUPPORT' if not valid_sidebands else
                              'BCD_CONTAMINATION_AND_PROTECTED_A_OVERLAP' if not purity_pass and not geometry_safe else
                              'BCD_CONTAMINATION_AT_OR_ABOVE_10_PERCENT' if not purity_pass else
                              'PROTECTED_A_OVERLAP')
        scan.append(row)
    scan = finalize_sideband_tradeoff_pd.DataFrame(scan)
    scan['mc_rank_all'] = scan.groupby('strategy').z_known_b_ratio_WP2_median.rank(ascending=False, method='min').astype(int)
    scan['mc_rank_purity'] = finalize_sideband_tradeoff_np.nan
    scan.loc[scan.purity_pass, 'mc_rank_purity'] = scan.loc[scan.purity_pass].groupby('strategy').z_known_b_ratio_WP2_median.rank(ascending=False, method='min')
    scan['pareto_purity_frontier'] = False
    scan['pareto_operational_frontier'] = False
    for strategy in ('loose', 'gap'):
        mask = scan.strategy.eq(strategy)
        scan.loc[mask, 'pareto_purity_frontier'] = finalize_sideband_tradeoff__pareto(scan.loc[mask], 'purity_pass')
        scan.loc[mask, 'pareto_operational_frontier'] = finalize_sideband_tradeoff__pareto(scan.loc[mask], 'operational_pass')
    selection = finalize_sideband_tradeoff_select_sideband_candidates(scan)
    validation.extend([
        dict(scope='Inputs', check='Exactly 154 nominal background coordinates and 120 separate signals per coordinate; no DATA sources', passed=True),
        dict(scope='Geometry', check='Analytic protected-A union retained; every loose/gap candidate has four disjoint rectangles', passed=True),
        dict(scope='Shortlist', check='Diagnostic shortlist selected using nominal MC only, before VR DATA closure', passed=True),
    ])
    metadata = dict(channel='2mu2e', main_wp='WP2', main_A=dict(iso_upper=.2, dphi_lower=2.7),
                    nominal_selection='DD, MJJ>=150 with overflow; partner isolation [0,0.5).',
                    original_data_files_opened=0, nominal_data_evaluated=False,
                    signal_hypotheses=120, sensitivity_hypotheses=60,
                    loose_candidates=64, gap_candidates=64, baseline_rows=3,
                    contamination_rule='All 120 hypotheses must have S/B_MC<0.10 in B, C and D individually; positive regional MC support >1e-8.',
                    protected_union='(iso<0.3 and dphi>=2.6) OR (iso<0.2 and dphi>=2.4); original reference A plus five historical candidate A regions.',
                    significance_scope='Same-model A-based known-background and finite-MC diagnostics only; closure and signal leakage are not a complete likelihood.',
                    gap_significance='A and its MC sensitivity are unchanged for every gap; changing BCD can alter background-estimate precision and full-inference sensitivity.',
                    closure_scope='Nominal MC kappa and VR validation are diagnostics, not a DATA-driven ranking or an assumed DD correction.',
                    statistical_scope='sumw/sumw2 first-order propagated ratios; N_eff and sideband_effective_precision are not observed raw event counts.',
                    shortlist_rule='Include WP2; nearest and highest-MC-sensitivity purity-passing loose definitions; best geometry-safe loose definition; nearest geometry-safe pure gap, one additional angle-gap step, and smallest MC prediction-relative-error gap. Deduplicate candidates.',
                    data_access_rule='operational_pass is an MC/geometry eligibility flag only; no nominal DATA loader or access approval is implemented.',
                    input_basis='Existing nominal grid regions are composed exactly into rectangular gap definitions; no interpolation or background floor.')
    return dict(tables=dict(sideband_scan=scan, sideband_regions=finalize_sideband_tradeoff_pd.DataFrame(regions),
                           sideband_signals=finalize_sideband_tradeoff_pd.DataFrame(signals), sideband_geometry=finalize_sideband_tradeoff_pd.DataFrame(geometry),
                           sideband_selection=selection, sideband_validation=finalize_sideband_tradeoff_pd.DataFrame(validation),
                           sideband_input_manifest=finalize_sideband_tradeoff_pd.DataFrame(manifest)), metadata=metadata)


def finalize_sideband_tradeoff__read(path, manifest):
    manifest.append(finalize_sideband_tradeoff_common.candidate_input_record(path, 'Sideband nominal MC cache only'))
    return finalize_sideband_tradeoff_pd.read_csv(path, float_precision='round_trip')


def finalize_sideband_tradeoff__coordinate_name(value):
    return f'{value:g}'.replace('.', 'p')


def finalize_sideband_tradeoff_candidate_specifications(background):
    iso = sorted(float(x) for x in background.tx.unique() if x >= .2)
    angle = sorted((float(y) for y in background.ty.unique() if y <= 2.7), reverse=True)
    if len(iso) != 8 or len(angle) != 8:
        raise AssertionError('Expected eight existing isolation and eight angle boundaries per approach.')
    rows = []
    for name, x, y in finalize_sideband_tradeoff_BASELINES:
        rows.append(dict(candidate_id=name, id=name, strategy='baseline', a_iso=x, a_phi=y,
                         b_iso=x, c_phi=y, display_label=f'{name} ({x:g}, {y:g})', native_steps=0))
    for strategy in ('loose', 'gap'):
        for ix, x in enumerate(iso):
            for iy, y in enumerate(angle):
                name = ('L' if strategy == 'loose' else 'G') + f'_i{finalize_sideband_tradeoff__coordinate_name(x)}_p{finalize_sideband_tradeoff__coordinate_name(y)}'
                a_iso, a_phi = (x, y) if strategy == 'loose' else (.2, 2.7)
                rows.append(dict(candidate_id=name, id=name, strategy=strategy, a_iso=a_iso,
                                 a_phi=a_phi, b_iso=x, c_phi=y, native_steps=ix+iy,
                                 display_label=(f'Loose ({x:g}, {y:g})' if strategy == 'loose'
                                                else f'Gap B iso≥{x:g}, C phi<{y:g}')))
    for spec in rows:
        rectangles = finalize_sideband_tradeoff_region_rectangles(spec)
        for i, first in enumerate(finalize_sideband_tradeoff_REGIONS):
            for second in finalize_sideband_tradeoff_REGIONS[i+1:]:
                if finalize_sideband_tradeoff__overlap(rectangles[first], rectangles[second]):
                    raise AssertionError('Candidate ABCD rectangles overlap.')
    return rows


def finalize_sideband_tradeoff_candidate_coordinates(spec):
    """Return the cached ordinary-WP coordinate supplying each gap rectangle."""
    if spec['b_iso'] < spec['a_iso'] or spec['c_phi'] > spec['a_phi']:
        raise ValueError('A gap requires b_iso>=a_iso and c_phi<=a_phi.')
    return dict(A=(float(spec['a_iso']), float(spec['a_phi'])),
                B=(float(spec['b_iso']), float(spec['a_phi'])),
                C=(float(spec['a_iso']), float(spec['c_phi'])),
                D=(float(spec['b_iso']), float(spec['c_phi'])))


def finalize_sideband_tradeoff_statistics_from_regions(values, variances, min_neff=10.):
    """Disjoint-region weighted statistics, without a denominator floor."""
    v = {r: float(values[r]) for r in finalize_sideband_tradeoff_REGIONS}
    w = {r: float(variances[r]) for r in finalize_sideband_tradeoff_REGIONS}
    if not finalize_sideband_tradeoff_np.isfinite(list(v.values()) + list(w.values())).all() or any(x < 0 for x in w.values()):
        raise ValueError('Regional yields and variances must be finite; variances nonnegative.')
    row = dict(pred=finalize_sideband_tradeoff_np.nan, pred_error=finalize_sideband_tradeoff_np.nan, kappa=finalize_sideband_tradeoff_np.nan, kappa_error=finalize_sideband_tradeoff_np.nan,
               pull=finalize_sideband_tradeoff_np.nan, status='UNDEFINED_PREDICTION', kappa_status='UNDEFINED_PREDICTION')
    for r in finalize_sideband_tradeoff_REGIONS:
        row.update({r: v[r], 'var_'+r: w[r], r+'_error': finalize_sideband_tradeoff_np.sqrt(w[r]),
                    'neff_'+r: v[r]**2/w[r] if w[r] > 0 else finalize_sideband_tradeoff_np.nan})
    neffs = finalize_sideband_tradeoff_np.asarray([row['neff_'+r] for r in finalize_sideband_tradeoff_REGIONS])
    row.update(neff_min=float(neffs.min()), min_neff=float(neffs.min()),
               min_sideband_neff=float(neffs[1:].min()),
               low_stat=bool(not finalize_sideband_tradeoff_np.all(finalize_sideband_tradeoff_np.isfinite(neffs) & (neffs >= min_neff))))
    a, b, c, d = (v[r] for r in finalize_sideband_tradeoff_REGIONS)
    if d <= 0:
        row.update(status='NONPOSITIVE_D', kappa_status='NONPOSITIVE_D')
        return row
    pred = b*c/d
    vp = (c/d)**2*w['B'] + (b/d)**2*w['C'] + (b*c/d**2)**2*w['D']
    row.update(pred=pred, pred_error=finalize_sideband_tradeoff_np.sqrt(vp),
               status='NONPOSITIVE_B_OR_C' if b <= 0 or c <= 0 else
               'LOW_STAT_SIDEBAND' if not finalize_sideband_tradeoff_np.all(finalize_sideband_tradeoff_np.isfinite(neffs[1:]) & (neffs[1:] >= min_neff)) else 'OK')
    if pred > 0:
        row['kappa'] = a/pred
        if a > 0:
            row['kappa_error'] = finalize_sideband_tradeoff_np.sqrt(w['A']/pred**2 + a**2*vp/pred**4)
            row['kappa_status'] = 'LOW_STAT' if row['low_stat'] else 'OK'
        else:
            row['kappa_status'] = 'NONPOSITIVE_A_ERROR_UNDEFINED'
    else:
        row['kappa_status'] = 'NONPOSITIVE_PREDICTION'
    if w['A']+vp > 0:
        row['pull'] = (a-pred)/finalize_sideband_tradeoff_np.sqrt(w['A']+vp)
    return row


def finalize_sideband_tradeoff_region_rectangles(spec):
    """Half-open rectangles: x underflow excluded, both angle flows retained."""
    finalize_sideband_tradeoff_candidate_coordinates(spec)
    return dict(A=(0., spec['a_iso'], spec['a_phi'], finalize_sideband_tradeoff_np.inf),
                B=(spec['b_iso'], finalize_sideband_tradeoff_np.inf, spec['a_phi'], finalize_sideband_tradeoff_np.inf),
                C=(0., spec['a_iso'], -finalize_sideband_tradeoff_np.inf, spec['c_phi']),
                D=(spec['b_iso'], finalize_sideband_tradeoff_np.inf, -finalize_sideband_tradeoff_np.inf, spec['c_phi']))


def finalize_sideband_tradeoff__overlap(left, right):
    return bool(max(left[0], right[0]) < min(left[1], right[1])
                and max(left[2], right[2]) < min(left[3], right[3]))


def finalize_sideband_tradeoff_region_geometry(spec, region):
    rectangle = finalize_sideband_tradeoff_region_rectangles(spec)[region]
    overlaps = [finalize_sideband_tradeoff__overlap(rectangle, (0., x, y, finalize_sideband_tradeoff_np.inf)) for x, y in finalize_sideband_tradeoff_PROTECTED]
    return dict(iso_low=rectangle[0], iso_high=rectangle[1], phi_low=rectangle[2], phi_high=rectangle[3],
                overlap_original_A=overlaps[0], overlap_protected_A=any(overlaps),
                region_disjoint_from_protected_A=not any(overlaps),
                nominal_data_evaluated=False)


def finalize_sideband_tradeoff__quantiles(row, field, values):
    values = finalize_sideband_tradeoff_pd.Series(values).replace([finalize_sideband_tradeoff_np.inf, -finalize_sideband_tradeoff_np.inf], finalize_sideband_tradeoff_np.nan).dropna()
    row[field+'_valid'] = len(values)
    row[field+'_median'] = float(values.median())
    row[field+'_q10'] = float(values.quantile(.1))
    row[field+'_q90'] = float(values.quantile(.9))


def finalize_sideband_tradeoff__pareto(frame, gate):
    output = finalize_sideband_tradeoff_pd.Series(False, index=frame.index)
    eligible = frame.loc[frame[gate] & finalize_sideband_tradeoff_np.isfinite(frame.z_known_b_ratio_WP2_median)
                         & finalize_sideband_tradeoff_np.isfinite(frame.prediction_relative_error)]
    for index, row in eligible.iterrows():
        dominated = ((eligible.z_known_b_ratio_WP2_median >= row.z_known_b_ratio_WP2_median-1e-12)
                     & (eligible.prediction_relative_error <= row.prediction_relative_error+1e-12)
                     & (eligible.max_S_B_BCD <= row.max_S_B_BCD+1e-12)
                     & ((eligible.z_known_b_ratio_WP2_median > row.z_known_b_ratio_WP2_median+1e-12)
                        | (eligible.prediction_relative_error < row.prediction_relative_error-1e-12)
                        | (eligible.max_S_B_BCD < row.max_S_B_BCD-1e-12)))
        output.loc[index] = not dominated.any()
    return output


def finalize_sideband_tradeoff_select_sideband_candidates(scan):
    """Keep a small set of distinct MC tradeoffs without looking at VR DATA."""
    chosen = []
    def add(frame, role, reason):
        if len(frame) == 0:
            return
        item = frame.iloc[0].to_dict()
        previous = next((r for r in chosen if r['candidate_id'] == item['candidate_id']), None)
        if previous is not None:
            previous['selection_role'] += '; '+role
            previous['selection_reason'] += '; '+reason
        else:
            item.update(selection_role=role, selection_reason=reason)
            chosen.append(item)
    add(scan.loc[scan.candidate_id.eq('WP2')], 'WP2 baseline', 'Main tight-A benchmark; purity and geometry failures remain visible.')
    loose = scan.loc[scan.strategy.eq('loose') & scan.purity_pass]
    add(loose.sort_values(['native_steps', 'z_known_b_ratio_WP2_median'], ascending=[True, False]),
        'Nearest purity-passing loose boundary', 'Smallest native-grid displacement; geometric protection is audited separately.')
    add(loose.sort_values(['z_known_b_ratio_WP2_median', 'prediction_relative_error'], ascending=[False, True]),
        'Highest-sensitivity pure loose boundary', 'Highest paired known-background MC sensitivity among BCD-pure loose candidates; may overlap protected A.')
    safe_loose = loose.loc[loose.geometry_safe]
    add(safe_loose.sort_values(['z_known_b_ratio_WP2_median', 'prediction_relative_error'], ascending=[False, True]),
        'Highest-sensitivity protected-safe loose boundary', 'Best MC sensitivity among loose candidates passing both purity and the existing geometric protection.')
    gap = scan.loc[scan.strategy.eq('gap') & scan.operational_pass]
    closest = gap.sort_values(['native_steps', 'b_iso', 'c_phi'], ascending=[True, True, False])
    add(closest, 'Smallest eligible gap', 'Nearest native-grid gap passing all BCD purity and protected-A tests; A remains WP2.')
    if len(closest):
        first = closest.iloc[0]
        additional = gap.loc[gap.b_iso.eq(first.b_iso) & gap.c_phi.lt(first.c_phi)].sort_values('c_phi', ascending=False)
        add(additional, 'Additional angle-gap step', 'One further available angle-gap step illustrates contamination margin versus remaining support.')
    add(gap.sort_values(['prediction_relative_error', 'native_steps'], ascending=[True, True]),
        'Best MC sideband precision gap', 'Smallest first-order relative BC/D error among eligible gaps; no claim of improved A-based significance.')
    result = finalize_sideband_tradeoff_pd.DataFrame(chosen)
    result.insert(0, 'selection_order', finalize_sideband_tradeoff_np.arange(1, len(result)+1))
    return result


# ========================================================================
# abcd_wp_comparison: original study API
# ========================================================================
"""Fixed MC-selected working points with identical nominal-SR and VR2 boundaries.

The reference calculation is loaded once. Each requested working point receives
its own closure, one-dimensional scans, composition and correlation tables. No
working-point ranking or nominal DATA projection is performed here.
"""
from copy import deepcopy as finalize_wp_comparison_deepcopy
from datetime import datetime as finalize_wp_comparison_datetime; from datetime import timezone as finalize_wp_comparison_timezone
import json as finalize_wp_comparison_json
from pathlib import Path as finalize_wp_comparison_Path

import matplotlib.pyplot as finalize_wp_comparison_plt
from matplotlib.backends.backend_pdf import PdfPages as finalize_wp_comparison_PdfPages
import numpy as finalize_wp_comparison_np
import pandas as finalize_wp_comparison_pd

finalize_wp_comparison_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_wp_comparison_plots = _FINALIZE_NAMESPACES['abcd_finalize_plots']
finalize_wp_comparison_review = _FINALIZE_NAMESPACES['review_mc_sr_working_points']


finalize_wp_comparison_REFERENCE_WP = (.30, 2.6)
finalize_wp_comparison_WORKING_POINTS = {
    '2mu2e': ((.20, 2.6), (.20, 2.7), (.20, 2.8)),
    '4mu': ((.30, 2.4), (.25, 2.4), (.40, 2.4)),
}


def finalize_wp_comparison__point_rows(frame, tx, ty):
    return frame.loc[finalize_wp_comparison_np.isclose(frame.tx, tx, rtol=0, atol=1e-12)
                     & finalize_wp_comparison_np.isclose(frame.ty, ty, rtol=0, atol=1e-12)].copy()


def finalize_wp_comparison__sr_view(reference, tx, ty):
    spec = dict(reference['specs']['SR'], tx=tx, ty=ty)
    tables = dict(reference['tables'])
    grid = tables['grid']
    indexed = grid.set_index(['tx', 'ty'], drop=False)
    coordinates, _ = finalize_wp_comparison_common.candidate_coordinates(spec, reference['planes']['SR']['SR'])
    tables['scan'] = finalize_wp_comparison_pd.DataFrame([
        dict(indexed.loc[(point['tx'], point['ty'])], scan_axis=point['scan_axis'],
             threshold=point['threshold']) for point in coordinates])
    tables['closure'] = finalize_wp_comparison__point_rows(grid, tx, ty)
    tables['purity_regions'] = finalize_wp_comparison__point_rows(tables['purity_grid_regions'], tx, ty)
    tables['selection'] = finalize_wp_comparison_pd.DataFrame([dict(plane='SR', **spec)])
    return dict(specs={'SR': spec}, planes=reference['planes'],
                selected=tables['selection'][['plane', 'tx', 'ty']].copy(), tables=tables)


def finalize_wp_comparison__validate_view(view, reference, config, wp):
    records = []
    tx, ty = view['selected'][['tx', 'ty']].iloc[0]
    metrics = (list('ABCD') + ['var_' + r for r in 'ABCD']
               + ['pred', 'pred_error', 'kappa', 'kappa_error'])
    if set(view['specs']) != {'PP', 'DP', 'PD'}:
        raise AssertionError('VR2 permits only PP, DP and PD.')
    finalize_wp_comparison_pd.testing.assert_frame_equal(view['tables']['grid'], reference['tables']['grid'])
    for key, spec in view['specs'].items():
        if (spec['state0'] + spec['state1'] != key
                or (spec['mass_low'], spec['mass_high']) != (150., 300.)
                or (spec['partner_iso_low'], spec['partner_iso_high']) != (0., .5)
                or (spec['tx'], spec['ty']) != (tx, ty)):
            raise AssertionError('A selected VR2 working point changed the shared cuts.')
        finalize_wp_comparison_common.candidate_coordinates(spec, view['planes'][key]['VR'])
        for source in ('MC', 'DATA'):
            fixed = view['tables']['closure'].query('plane == @key and source == @source')
            cached = finalize_wp_comparison__point_rows(reference['tables']['grid'].query(
                'plane == @key and source == @source'), tx, ty)
            if len(fixed) != 1 or len(cached) != 1:
                raise AssertionError('Each selected WP requires exactly one closure row.')
            finalize_wp_comparison_np.testing.assert_allclose(fixed[metrics], cached[metrics], equal_nan=True,
                                       rtol=1e-10, atol=1e-8)
            if source == 'DATA' and not bool(fixed.iloc[0].approved):
                if not fixed[metrics].isna().all().all() or view['data_planes'][key] is not None:
                    raise AssertionError('Failed DATA gate must hide the selected point.')
            records.append(dict(wp=wp, plane=key, source=source, tx=tx, ty=ty,
                                check='Selected closure equals validated full grid', passed=True))
    for context, tables in [('VR', view['tables']), ('SR', view['sr_reference']['tables'])]:
        scan = tables['scan']
        if (not finalize_wp_comparison_np.isclose(scan.loc[scan.scan_axis.eq('x'), 'ty'], ty).all()
                or not finalize_wp_comparison_np.isclose(scan.loc[scan.scan_axis.eq('y'), 'tx'], tx).all()):
            raise AssertionError('A scan retained the reference fixed boundary.')
        records.append(dict(wp=wp, plane=context, source='MC / approved VR DATA', tx=tx, ty=ty,
                            check='Scans hold the other selected WP boundary fixed', passed=True))
    sr = view['sr_reference']
    if (sr['specs']['SR']['tx'], sr['specs']['SR']['ty']) != (tx, ty):
        raise AssertionError('Nominal SR and VR boundaries must be identical.')
    if (set(sr['tables']['closure'].source) != {'MC'} or 'data_planes' in sr
            or set(sr['planes']) != {'SR'}):
        raise AssertionError('The nominal SR view must contain only MC.')
    independent = finalize_wp_comparison_common.plane_statistics(sr['planes']['SR']['SR'], tx, ty,
                                          min_neff=config['min_neff'])
    fixed = sr['tables']['closure'].iloc[0]
    finalize_wp_comparison_np.testing.assert_allclose([independent[name] for name in metrics], fixed[metrics].to_numpy(float),
                               equal_nan=True, rtol=1e-10, atol=1e-8)
    blocked = view['tables']['grid'].query('source == "DATA" and not approved')
    if not blocked[metrics].isna().all().all():
        raise AssertionError('Failed DATA grid points must remain hidden.')
    records.append(dict(wp=wp, plane='SR', source='MC', tx=tx, ty=ty,
                        check='Same SR/VR WP; direct MC partition; no nominal DATA', passed=True))
    return finalize_wp_comparison_pd.DataFrame(records)


def finalize_wp_comparison__working_point_view(reference, config, wp, tx, ty):
    specs = {key: dict(spec, tx=tx, ty=ty) for key, spec in reference['specs'].items()}
    gate_columns = ['approved', 'purity_pass', 'max_S_B', 'max_S_B_error', 'worst_sample',
                    'worst_region', 'signal_hypotheses', 'valid_background', 'gate_reason']
    mc_grid = reference['tables']['grid'].loc[reference['tables']['grid'].source.eq('MC')]
    gates = {(row['plane'], row['tx'], row['ty']): {field: row[field] for field in gate_columns}
             for row in mc_grid.to_dict('records')}
    for key, plane in reference['data_planes'].items():
        if plane is None and any(gate['approved'] for (name, _, _), gate in gates.items() if name == key):
            raise ValueError(f'{key}: reference display does not retain an approved VR DATA plane.')
    tables = dict(reference['tables'])
    # Recompute WP-dependent quantities from the already loaded planes. The full
    # grid is required to stay identical, including every blocked DATA value.
    tables.update(finalize_wp_comparison_common.collect_candidate_tables(reference['planes'], reference['data_planes'],
                                                   gates, specs, config))
    tables['selection'] = finalize_wp_comparison_pd.DataFrame([dict(plane=key, **spec) for key, spec in specs.items()])
    tables['purity_regions'] = finalize_wp_comparison__point_rows(tables['purity_grid_regions'], tx, ty)
    tables['summary'] = finalize_wp_comparison_common.candidate_summary(tables, specs)
    sr = finalize_wp_comparison__sr_view(reference['sr_reference'], tx, ty)
    tables.update({'sr_reference_' + name: frame for name, frame in sr['tables'].items()})
    definitions = finalize_wp_comparison_deepcopy(reference['definitions'])
    definitions.update(definitions=specs, working_point=wp,
                       selection_origin='Previously fixed nominal MC SR candidates; no VR DATA ranking')
    view = dict(specs=specs, definitions=definitions,
                selected=tables['selection'][['plane', 'tx', 'ty']].copy(), planes=reference['planes'],
                data_planes={key: plane if gates[key, tx, ty]['approved'] else None
                             for key, plane in reference['data_planes'].items()},
                distribution_purity=reference['distribution_purity'], tables=tables, sr_reference=sr)
    tables['validation'] = finalize_wp_comparison__validate_view(view, reference, config, wp)
    return view


def finalize_wp_comparison_run_wp_comparison(config, working_points):
    """Load once and compare exactly the three previously chosen channel WPs."""
    finalize_wp_comparison_common.validate_finalize_config(config)
    chosen = tuple(tuple(float(value) for value in point) for point in working_points)
    if config['vr'] != 'VR2' or chosen != finalize_wp_comparison_WORKING_POINTS[config['topology']]:
        raise ValueError('Use the three fixed MC-selected WPs for this channel in VR2.')
    formula_validation = finalize_wp_comparison_review.validate_significance()
    reference = finalize_wp_comparison_common.run_candidate_comparison(config)
    original_selection = reference['selected'].copy(deep=True)
    original_specs = finalize_wp_comparison_deepcopy(reference['specs'])
    points = {f'WP{index}': finalize_wp_comparison__working_point_view(reference, config, f'WP{index}', *point)
              for index, point in enumerate(chosen, 1)}
    finalize_wp_comparison_pd.testing.assert_frame_equal(reference['selected'], original_selection)
    if reference['specs'] != original_specs:
        raise AssertionError('Building candidate views changed the reference specifications.')
    coordinates = finalize_wp_comparison_pd.DataFrame([dict(wp='Reference', tx=finalize_wp_comparison_REFERENCE_WP[0], ty=finalize_wp_comparison_REFERENCE_WP[1])]
                               + [dict(wp=wp, tx=point[0], ty=point[1])
                                  for wp, point in zip(points, chosen)])
    sr_tables = reference['sr_reference']['tables']
    full_significance = finalize_wp_comparison_review.signal_significance(config['topology'], sr_tables['grid'],
                                                  sr_tables['signal_points'])
    significance = coordinates.merge(full_significance, on=['tx', 'ty'], validate='one_to_many')
    summary = finalize_wp_comparison_review.summarize_significance(significance)
    summary = coordinates.merge(summary, on=['tx', 'ty'], validate='one_to_many')
    primary = summary.loc[summary.signal_set.eq('matching_decay')].reset_index(drop=True)
    if len(primary) != 4 or not primary.hypotheses.eq(60).all():
        raise AssertionError('Each WP needs 60 separate matching-decay significance hypotheses.')
    views = dict(Reference=reference, **points)
    vr_closure = finalize_wp_comparison_pd.concat([view['tables']['closure'].assign(wp=wp) for wp, view in views.items()],
                           ignore_index=True)
    sr_closure = finalize_wp_comparison_pd.concat([view['sr_reference']['tables']['closure'].assign(wp=wp)
                           for wp, view in views.items()], ignore_index=True)
    bins = reference['tables']['distribution_bin_purity']
    blinding = []
    for wp, view in views.items():
        for key in config['candidate_order']:
            plane_bins = bins.loc[bins.plane.eq(key)]
            allowed = bool(view['tables']['closure'].query(
                'plane == @key and source == "DATA"').iloc[0].approved)
            supported = plane_bins.valid_background.eq(True)
            blinding.append(dict(wp=wp, plane=key, tx=view['specs'][key]['tx'],
                                 ty=view['specs'][key]['ty'], integrated_gate_approved=allowed,
                                 displayed_bins=len(plane_bins),
                                 approved_bins=int(plane_bins.approved.sum()) if allowed else 0,
                                 blinded_bins=int((~plane_bins.approved).sum()) if allowed else len(plane_bins),
                                 contamination_blinded_bins=int((supported & plane_bins.max_S_B.ge(.1)).sum()),
                                 unsupported_background_bins=int((~supported).sum())))
    validation = finalize_wp_comparison_pd.concat([view['tables']['validation'] for view in points.values()], ignore_index=True)
    extra = [dict(wp='All', plane='SR', source='MC', check=check, passed=passed)
             for check, passed in formula_validation.items()]
    extra.append(dict(wp='Reference', plane='VR / SR', source='MC / approved VR DATA',
                      check='Reference selection and specifications unchanged', passed=True))
    validation = finalize_wp_comparison_pd.concat([validation, finalize_wp_comparison_pd.DataFrame(extra)], ignore_index=True)
    tables = dict(working_points=coordinates.copy(), significance_summary=primary,
                  significance_summary_all_sets=summary, signal_significance=significance,
                  vr_closure=vr_closure, sr_closure=sr_closure, validation=validation,
                  distribution_blinding_summary=finalize_wp_comparison_pd.DataFrame(blinding))
    return dict(reference=reference, points=points, working_points=coordinates, tables=tables)


def finalize_wp_comparison__summary_figures(result, config):
    summary = result['tables']['significance_summary']
    positions = finalize_wp_comparison_np.arange(len(summary))
    labels = [f'{row.wp}\n({row.tx:g}, {row.ty:g})' for row in summary.itertuples()]
    channel = finalize_wp_comparison_plots._opt_channel(config)
    figures = []
    with finalize_wp_comparison_plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig, axes = finalize_wp_comparison_plt.subplots(1, 2, figsize=(13, 5.7))
        for ax, metric, title, color in zip(axes,
                ('z_known_b', 'z_mc_stat_diagnostic'),
                ('Known-background Asimov Z', 'MC-statistical auxiliary diagnostic'),
                ('#0072B2', '#D55E00')):
            middle = summary[metric + '_ratio_median'].to_numpy(float)
            lower = summary[metric + '_ratio_q10'].to_numpy(float)
            upper = summary[metric + '_ratio_q90'].to_numpy(float)
            ax.errorbar(positions, middle, yerr=finalize_wp_comparison_np.array([middle - lower, upper - middle]),
                        fmt='o', color=color, capsize=4, label='Median and 10–90% model quantiles')
            ax.axhline(1, color='#444444', lw=1, ls='--')
            for x, value in zip(positions, middle):
                ax.annotate(f'{value:.4g}', (x, value), xytext=(6, 5), textcoords='offset points')
            ax.set(xticks=positions, xticklabels=labels, ylabel='Z(WP) / Z(reference)', title=title)
            ax.grid(alpha=.25)
        fig.suptitle(f'{channel} · nominal DD MC SR significance at fixed candidates', fontsize=13)
        fig.text(.5, .015, '60 matching-decay hypotheses are evaluated separately; bars show model spread, not statistical errors.\n'
                 'Production signal normalization is retained. The MC-statistical variant is a diagnostic, not a fitted ABCD significance.\n'
                 'M(LJ,LJ) ≥ 150 GeV; partner isolation < 0.5. No nominal DATA is used; candidates are not re-ranked here.',
                 ha='center', va='bottom', fontsize=8.5)
        fig.subplots_adjust(top=.84, bottom=.25, left=.075, right=.98, wspace=.27)
        figures.append((fig, 'summary_nominal_mc_significance'))

        fig, axes = finalize_wp_comparison_plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].errorbar(positions, summary.b, yerr=summary.sigma_b, fmt='o', color='#202020', capsize=4)
        axes[0].set_ylabel('Nominal SR background MC yield ± √sumw²')
        axes[1].plot(positions, summary.neff_b, 'o', color='#0072B2')
        axes[1].axhline(10, color='#888888', ls='--', label='N_eff = 10 diagnostic')
        axes[1].set_ylabel('Nominal SR background N_eff = (sumw)² / sumw²')
        axes[1].legend(fontsize=9)
        for ax, column in zip(axes, ('b', 'neff_b')):
            ax.set(xticks=positions, xticklabels=labels)
            ax.grid(alpha=.25)
            for x, value in zip(positions, summary[column]):
                ax.annotate(f'{value:.4g}', (x, value), xytext=(6, 5), textcoords='offset points')
        fig.suptitle(f'{channel} · nominal DD MC SR background support', fontsize=13)
        fig.text(.5, .015, 'Low effective MC counts limit the stability of significance comparisons; no background floor is imposed.\n'
                 'Uncertainties are production sumw²; no closure or detector systematic uncertainty is included.',
                 ha='center', va='bottom', fontsize=8.5)
        fig.subplots_adjust(top=.84, bottom=.23, left=.09, right=.97, wspace=.30)
        figures.append((fig, 'summary_nominal_mc_background'))

        fig, axes = finalize_wp_comparison_plt.subplots(1, 3, figsize=(18, 6))
        closure = result['tables']['vr_closure']
        for ax, key in zip(axes, config['candidate_order']):
            ax.axhspan(.9, 1.1, color='#009E73', alpha=.12)
            ax.axhline(1, color='#777777', ls='--', lw=1)
            for source, offset, color in [('MC', -.08, '#E69F00'), ('DATA', .08, '#202020')]:
                frame = closure.query('plane == @key and source == @source').set_index('wp').loc[summary.wp]
                x = positions + offset
                finalize_wp_comparison_plots._opt_reference_errorbar(ax, x, frame.kappa.to_numpy(float),
                                              error=frame.kappa_error.to_numpy(float),
                                              fmt='o', color=color, capsize=3, label=source)
                finalize_wp_comparison_plots._opt_closure_limits(ax, x, frame.kappa.to_numpy(float),
                                         frame.kappa_error.to_numpy(float),
                                         note_y=.93 if source == 'MC' else .84)
            ax.set(title=f'VR2 {key}', xticks=positions, xticklabels=labels, ylabel='κ = A / (BC/D)', ylim=(0, 2))
            ax.grid(alpha=.25)
            ax.legend(loc='upper left', fontsize=9)
        fig.suptitle(f'{channel} · identical boundaries in VR2 PP / DP / PD', fontsize=13)
        fig.text(.5, .015, '150 ≤ M(LJ,LJ) < 300 GeV; partner isolation < 0.5. Each candidate uses the same boundaries as nominal MC SR.\n'
                 'DATA require the integrated all-120-hypothesis S/B < 10% gate. Green shading is only a κ = 0.9–1.1 guide.\n'
                 'Nearby working points reuse events; errors are correlated across WPs. Low-count symmetric errors are approximate.',
                 ha='center', va='bottom', fontsize=8.5)
        fig.subplots_adjust(top=.84, bottom=.24, left=.055, right=.98, wspace=.27)
        figures.append((fig, 'summary_vr_closure'))
    return figures


def finalize_wp_comparison_make_wp_figures(result, config):
    """Three overview figures followed by the existing 21 figures for each WP."""
    figures = finalize_wp_comparison__summary_figures(result, config)
    for wp, view in result['points'].items():
        point_figures = finalize_wp_comparison_plots.make_comparison_figures(view, config)
        for figure, stem in point_figures:
            if figure._suptitle is not None:
                figure._suptitle.set_text(wp + ' · ' + figure._suptitle.get_text())
            figures.append((figure, wp.lower() + '_' + stem))
            finalize_wp_comparison_plt.close(figure)
    for figure, _ in figures:
        finalize_wp_comparison_plt.close(figure)
    if len(figures) != 66 or len({stem for _, stem in figures}) != len(figures):
        raise AssertionError('Expected three overview figures and 21 unique figures per WP.')
    return figures


def finalize_wp_comparison_export_wp_comparison(result, config, figures):
    """Export fixed-WP views and shared grids without duplicating signal grids."""
    output = finalize_wp_comparison_Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for order, (figure, stem) in enumerate(figures, 1):
        png = stem + '.png'
        figure.savefig(output / png, dpi=100, bbox_inches='tight')
        records.append(dict(figure=stem, png=png, order=order))
    result['tables']['figure_manifest'] = finalize_wp_comparison_pd.DataFrame(records)
    for name, frame in result['tables'].items():
        frame.to_csv(output / f'{name}.csv', index=False)
    reference = result['reference']
    # Shared grids retain the complete approved/blocked scan record just once.
    canonical_exports = []
    canonical_dir = output.parent / {'2mu2e': '03_2mu2e_VR2_summary_outputs',
                                     '4mu': '04_4mu_VR2_summary_outputs'}[config['topology']]
    for name in ('grid', 'purity_grid_regions', 'distribution_bin_purity',
                 'sr_reference_grid', 'sr_reference_purity_grid_regions',
                 'sr_reference_regression', 'sr_reference_partition_validation'):
        path = output / f'shared_{name}.csv'
        canonical = canonical_dir / f'{name}.csv'
        csv_bytes = reference['tables'][name].to_csv(index=False).encode('utf-8')
        if (canonical.is_file() and canonical.read_bytes() == csv_bytes
                and (not path.exists() or path.read_bytes() == csv_bytes)):
            canonical_exports.append(dict(alias=path.name,
                canonical_path=str(canonical.resolve()), size_bytes=len(csv_bytes),
                sha256=finalize_wp_comparison_common._comparison_hash(canonical)))
            path.unlink(missing_ok=True)
        else:
            reference['tables'][name].to_csv(path, index=False)
    fixed_tables = ('selection', 'closure', 'scan', 'composition', 'correlations', 'purity_regions', 'summary')
    for wp, view in dict(Reference=reference, **result['points']).items():
        for name in fixed_tables:
            view['tables'][name].to_csv(output / f'{wp.lower()}_{name}.csv', index=False)
        for name in ('selection', 'closure', 'scan', 'purity_regions'):
            view['sr_reference']['tables'][name].to_csv(output / f'{wp.lower()}_sr_{name}.csv', index=False)
    sources = reference['tables']['input_manifest'].to_dict('records')
    for source in (finalize_wp_comparison_Path(__file__), finalize_wp_comparison_Path(finalize_wp_comparison_review.__file__)):
        sources.append(finalize_wp_comparison_common.candidate_input_record(source, 'Fixed-WP implementation'))
    finalize_wp_comparison_pd.DataFrame(sources).to_csv(output / 'input_manifest.csv', index=False)
    manifest = dict(created_utc=finalize_wp_comparison_datetime.now(finalize_wp_comparison_timezone.utc).isoformat(), settings=config,
                    working_points=result['working_points'].to_dict('records'),
                    reference_specs=reference['specs'],
                    candidate_specs={wp: view['specs'] for wp, view in result['points'].items()},
                    nominal_mc_sr_specs={wp: view['sr_reference']['specs']
                                         for wp, view in dict(Reference=reference, **result['points']).items()},
                    candidate_origin='Previously fixed MC SR shortlist; no ranking is rerun and VR DATA do not choose WPs.',
                    significance='Known-background Asimov Z=sqrt(2*((s+b)*log(1+s/b)-s)); s and b are nominal DD MC SR A yields. Each matching-decay hypothesis is evaluated separately (60 primary; all 120 exported individually). MC-statistical auxiliary variant uses variance=sumw2 as a diagnostic; it is not an ABCD likelihood or final discovery significance. No detector/closure systematics are included.',
                    significance_source=finalize_wp_comparison_review.SOURCE,
                    summary_bars='10th–90th percentiles across signal hypotheses, not statistical uncertainty intervals.',
                    export_storage='All 66 figures are embedded in the notebook and retained as vector pages in one combined PDF. Only the three summary PNG previews are additionally saved at 100 dpi; individual PDFs and WP PNG duplicates are omitted. Selected reference plus three WP signal yields/significances (4 x 120) are exported. Full 154-point signal grids are recalculated in memory but not duplicated here; original production inputs and implementation hashes remain in input_manifest.csv.',
                    data_access='Only VR2 [150,300) PP/DP/PD DATA planes are loaded. No nominal DD DATA projection or nominal DATA sideband scan is performed.',
                    purity_policy='Integrated DATA gates require max S/B_MC <0.10 separately in all ABCD regions across all 120 distinct hypotheses. Failed grid values remain NaN.',
                    distribution_blinding='Identical MC-derived exact displayed-bin mask for all WPs: all 120 signals must be finite/nonnegative, background >1e-8 and max S/B_MC <0.10 after display folding. The selected integrated WP gate is also required. Hidden bins do not set the DATA color scale. This display mask does not change integrated ABCD yields.',
                    scans='Native iso thresholds in (0,2] and dphi thresholds in (0,pi]; each one-dimensional scan fixes the other coordinate at its own selected WP. Full grids are shared, and all WP boundaries are identical between VR and nominal MC SR.',
                    uncertainty='Existing independent-ABCD sumw2 delta method; low-statistics and zero-sideband cases retain their explicit status flags.',
                    validation='Reference configuration and source tables validated by existing pipeline; each selected closure checked against the full grid; SR re-summed independently; scan fixed coordinates checked; no nominal DATA; baseline specifications preserved; significance formula limits tested.',
                    implementation_hashes={entry['path']: entry['sha256'] for entry in sources
                                           if entry['source'] in ('Analysis implementation', 'Fixed-WP implementation')},
                    figures=records,
                    export_maintenance=dict(action='reuse_byte_identical_canonical_csv',
                        canonical_exports=canonical_exports, physics_values_changed=False,
                        exporter_sha256=finalize_wp_comparison_common._comparison_hash(finalize_wp_comparison_Path(__file__))))
    (output / 'manifest.json').write_text(finalize_wp_comparison_json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + '\n')
    print('Saved:', output)


# ========================================================================
# abcd_4mu_final_mc: original study API
# ========================================================================
"""4μ loose-boundary/fixed-A buffer comparison from the established 6D MC.

Only the reconstructed 4μ channel is projected. All 120 generated signal
hypotheses remain separate; sensitivity summaries use the 60 matching 4Mu
hypotheses. This module never opens DATA and never selects a passing candidate
when no nominal B/C/D contamination gate passes.
"""
from pathlib import Path as finalize_4mu_final_mc_Path
import ast as finalize_4mu_final_mc_ast
import gc as finalize_4mu_final_mc_gc
import hashlib as finalize_4mu_final_mc_hashlib
import json as finalize_4mu_final_mc_json
import pickle as finalize_4mu_final_mc_pickle
import sys as finalize_4mu_final_mc_sys

import numpy as finalize_4mu_final_mc_np
import pandas as finalize_4mu_final_mc_pd

finalize_4mu_final_mc_SIDM_DIR = finalize_4mu_final_mc_Path(__file__).resolve().parents[2]
for finalize_4mu_final_mc_folder in (finalize_4mu_final_mc_SIDM_DIR.parent, finalize_4mu_final_mc_Path(__file__).resolve().parent):
    if str(finalize_4mu_final_mc_folder) not in finalize_4mu_final_mc_sys.path:
        finalize_4mu_final_mc_sys.path.insert(0, str(finalize_4mu_final_mc_folder))
finalize_4mu_final_mc_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_4mu_final_mc_review = _FINALIZE_NAMESPACES['review_mc_sr_working_points']
finalize_4mu_final_mc_statistics_from_regions = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].statistics_from_regions

finalize_4mu_final_mc_STATES = ('PP', 'PD', 'DP', 'DD')
finalize_4mu_final_mc_REGIONS = tuple('ABCD')
finalize_4mu_final_mc_LOOSE_ISOLATION = (.2, .25, .3, .4, .5)
finalize_4mu_final_mc_LOOSE_ANGLES = (2.4, 2.6, 2.7, 2.8)
finalize_4mu_final_mc_BUFFER_ISOLATION = (.25, .3, .4, .5, .75, 1., 2.)
finalize_4mu_final_mc_BUFFER_ANGLES = (.5, 1., 1.5, 2., 2.2, 2.4)
finalize_4mu_final_mc_REFERENCE = (.3, 2.6)
finalize_4mu_final_mc_CACHE_VERSION = 1


def finalize_4mu_final_mc_build_mc_study(notebook_dir):
    """Return 4μ-only projected MC, exact scans, and regression checks."""
    notebook_dir = finalize_4mu_final_mc_Path(notebook_dir).resolve()
    config = finalize_4mu_final_mc_json.loads((notebook_dir / '04_4mu_VR2_summary_outputs' / 'manifest.json').read_text())['settings']
    if (config['topology'], config['current_hist'], config['expected_current_channel'],
            config['partner_iso_low'], config['partner_iso_high'], config['purity_limit']) != (
            '4mu', 'abcd6d_4mu', 'abcd6d_inclusive_4mu_dxySpread_cosAlpha_mu_veto_eHEM', 0., .5, .1):
        raise ValueError('Retain the existing 4mu histogram, partner isolation and 10% conventions.')
    config = dict(config, notebook_dir=str(notebook_dir),
        output_dir=str(notebook_dir / '10_4mu_final_check_outputs'),
        reference_wp=list(finalize_4mu_final_mc_REFERENCE), priority_anchor='WP2',
        anchor_wp2=dict(a_iso=.25, a_phi=2.4), anchor_wp1=dict(a_iso=.3, a_phi=2.4),
        nominal_DATA_A_evaluated=False, selection_status='UNSELECTED_MC_STUDY')
    bkg_paths = sorted(finalize_4mu_final_mc_Path(config['bkg_dir']).glob('*.coffea'))
    config['signal_dir'] = finalize_4mu_final_mc_Path(config['signal_dir'])
    signal_paths = finalize_4mu_final_mc_common.candidate_signal_paths(config)
    config['signal_dir'] = str(config['signal_dir'])
    if len(bkg_paths) != 18:
        raise ValueError('Expected the established 18 background MC files.')
    records = [finalize_4mu_final_mc_common.candidate_input_record(path, '4mu background MC' if path in bkg_paths else
               'Generated hypothesis in reconstructed 4mu channel') for path in bkg_paths + signal_paths]
    fingerprint = finalize_4mu_final_mc_hashlib.sha256(finalize_4mu_final_mc_json.dumps(dict(version=finalize_4mu_final_mc_CACHE_VERSION, files=records,
        projection_implementation=finalize_4mu_final_mc__projection_signature(),
        common_implementation=finalize_4mu_final_mc_common._comparison_hash(finalize_4mu_final_mc_Path(finalize_4mu_final_mc_common.__file__))), sort_keys=True).encode()).hexdigest()
    cache = notebook_dir / '10_4mu_final_check_outputs' / 'cache' / 'mc_projected_4mu.pkl'
    reused = False
    if cache.is_file():
        with cache.open('rb') as handle:
            saved = finalize_4mu_final_mc_pickle.load(handle)
        if saved.get('fingerprint') == fingerprint:
            projected = saved['result']
            reused = True
    if not reused:
        projected = finalize_4mu_final_mc__load_mc(config, bkg_paths, signal_paths)
        projected['input_records'] = records
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.pkl.tmp')
        with temporary.open('wb') as handle:
            finalize_4mu_final_mc_pickle.dump(dict(fingerprint=fingerprint, result=projected), handle,
                        protocol=finalize_4mu_final_mc_pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache)
    candidates = finalize_4mu_final_mc_pd.DataFrame(finalize_4mu_final_mc_candidate_specifications())
    tables = finalize_4mu_final_mc__scan_tables(projected, candidates)
    tables['validation'] = finalize_4mu_final_mc__regression(tables, projected, notebook_dir)
    tables['mc_input_manifest'] = finalize_4mu_final_mc_pd.DataFrame(records)
    tables['candidates'] = candidates
    tables['mc_scopes'] = finalize_4mu_final_mc_pd.DataFrame(projected['scopes'].values())
    tables['mc_input_availability'] = finalize_4mu_final_mc_pd.DataFrame([
        dict(quantity='Selected raw entries', available=False,
             reason='Existing selected 6D histogram stores sumw and sumw2, not unweighted entries'),
        dict(quantity='Weighted ABCD yields and statistical variances', available=True,
             reason='Stored 6D Weight moments; production normalization is reused'),
        dict(quantity='Original DATA', available=False,
             reason='Never loaded by the MC study; nominal DATA require an independent permission audit'),
    ])
    nominal = tables['mc_scan'].loc[tables['mc_scan'].scope.eq('nominal_DD')]
    passes = nominal.loc[nominal.purity_pass_BCD]
    metadata = dict(topology='4mu', reconstructed_channel='4mu_only', priority_anchor='WP2',
        source_fingerprint=fingerprint, projected_cache_reused=reused,
        table_implementation_sha256=finalize_4mu_final_mc_common._comparison_hash(finalize_4mu_final_mc_Path(__file__)),
        original_background_files_opened=0 if reused else 18,
        original_signal_files_opened=0 if reused else 120, original_DATA_files_opened=0,
        nominal_DATA_A_evaluated=False, nominal_DATA_BCD_evaluated=False,
        generated_signal_hypotheses=120, matching_4Mu_hypotheses=60,
        nominal_candidate_count=len(nominal), nominal_BCD_gate_passing_candidates=len(passes),
        selected_candidate=None, candidate_selection_performed=False,
        selection_status='NO_NOMINAL_CONTROL_GATE_PASS' if passes.empty else 'MC_GATE_PASS_REQUIRES_DATA_AUDIT',
        raw_selected_entries_available=False, unchanged_production_normalization=True,
        reference_wp=finalize_4mu_final_mc_REFERENCE, partner_isolation=(0., .5),
        all_numeric_checks_passed=bool(tables['validation'].passed.all()),
        significance_interpretation='Per-model MC-A diagnostic; not a full ABCD likelihood or an observed significance',
        closure_interpretation='MC closure only; no VR-to-SR correction or nuisance has been adopted')
    return dict(config=config, tables=tables, backgrounds=projected['backgrounds'],
                signals=projected['signals'], scopes=projected['scopes'], metadata=metadata)


def finalize_4mu_final_mc_candidate_specifications():
    """Twenty native loose points and 78 fixed-A buffer geometries."""
    rows = []
    for isolation in finalize_4mu_final_mc_LOOSE_ISOLATION:
        for angle in finalize_4mu_final_mc_LOOSE_ANGLES:
            label = ('WP2' if finalize_4mu_final_mc_np.isclose(isolation, .25) and finalize_4mu_final_mc_np.isclose(angle, 2.4) else
                     'WP1' if finalize_4mu_final_mc_np.isclose(isolation, .3) and finalize_4mu_final_mc_np.isclose(angle, 2.4) else
                     'Reference' if finalize_4mu_final_mc_np.isclose(isolation, .3) and finalize_4mu_final_mc_np.isclose(angle, 2.6) else '')
            rows.append(dict(candidate_id=f'L_i{finalize_4mu_final_mc__tag(isolation)}_p{finalize_4mu_final_mc__tag(angle)}',
                strategy='loose', anchor='loose', anchor_label=label,
                a_iso=isolation, a_phi=angle, b_iso=isolation, c_phi=angle,
                scan_tier='main', is_main=True, priority_anchor=False,
                fixed_A=False, isolation_buffer_width=0., angular_buffer_width=0.))
    for anchor, isolation in (('WP2', .25), ('WP1', .3)):
        for high in finalize_4mu_final_mc_BUFFER_ISOLATION:
            if high < isolation:
                continue
            for low in finalize_4mu_final_mc_BUFFER_ANGLES:
                rows.append(dict(candidate_id=f'G_{anchor}_i{finalize_4mu_final_mc__tag(high)}_p{finalize_4mu_final_mc__tag(low)}',
                    strategy='buffer_'+anchor.lower(), anchor=anchor, anchor_label=anchor,
                    a_iso=isolation, a_phi=2.4, b_iso=high, c_phi=low,
                    scan_tier='main' if high <= .5 else 'wide', is_main=high <= .5,
                    priority_anchor=anchor == 'WP2', fixed_A=True,
                    isolation_buffer_width=high-isolation, angular_buffer_width=2.4-low))
    for spec in rows:
        spec['main_scan'] = spec['is_main']
        spec['in_primary_scan'] = spec['is_main']
    if len(rows) != 98:
        raise AssertionError('Expected 20 loose and 78 buffer candidates.')
    return rows


def finalize_4mu_final_mc_candidate_masks(plane, spec):
    """Disjoint rectangles with physical isolation and retained angle flows."""
    ai, ap, bi, cp = (float(spec[k]) for k in ('a_iso', 'a_phi', 'b_iso', 'c_phi'))
    if bi < ai or cp > ap:
        raise ValueError('A buffer must keep B/D outside A isolation and C/D below A angle.')
    for field, cut in (('xedges', ai), ('xedges', bi), ('yedges', ap), ('yedges', cp)):
        finalize_4mu_final_mc_common._comparison_split(plane[field], cut)
    x = finalize_4mu_final_mc_np.r_[-finalize_4mu_final_mc_np.inf, plane['xedges']][:, None]
    y = finalize_4mu_final_mc_np.r_[-finalize_4mu_final_mc_np.inf, plane['yedges']][None, :]
    return dict(A=(x >= 0) & (x < ai) & (y >= ap), B=(x >= bi) & (y >= ap),
                C=(x >= 0) & (x < ai) & (y < cp), D=(x >= bi) & (y < cp))


def finalize_4mu_final_mc_candidate_regions(plane, spec):
    """Exact weighted moments; no floors, clipping, or invented empty-bin errors."""
    return {r: dict(value=float(plane['values'][mask].sum()),
                    variance=float(plane['variances'][mask].sum()))
            for r, mask in finalize_4mu_final_mc_candidate_masks(plane, spec).items()}


def finalize_4mu_final_mc_project_cube(weights, variances, edges, state, mass_low=0., mass_high=None):
    """Pure array projection of an explicitly selected 4mu state and mass window."""
    if state not in finalize_4mu_final_mc_STATES:
        raise ValueError('Explicit leading/subleading PP, PD, DP, DD required.')
    first = finalize_4mu_final_mc_common._comparison_split(edges['disp0'], 2.5)
    second = finalize_4mu_final_mc_common._comparison_split(edges['disp1'], 2.5)
    index = (slice(None), slice(finalize_4mu_final_mc_common._comparison_split(edges['iso1'], 0.),
                                  finalize_4mu_final_mc_common._comparison_split(edges['iso1'], .5)),
             slice(first, None) if state[0] == 'P' else slice(None, first),
             slice(second, None) if state[1] == 'P' else slice(None, second),
             slice(None), slice(finalize_4mu_final_mc_common._comparison_split(edges['ljlj_mass'], mass_low),
                                None if mass_high is None else finalize_4mu_final_mc_common._comparison_split(edges['ljlj_mass'], mass_high)))
    arrays = [a[index].sum(axis=(1, 2, 3, 5)) for a in (weights, variances)]
    return finalize_4mu_final_mc_common._comparison_plane(*arrays, edges['iso0'], edges['abs_dphi'])


def finalize_4mu_final_mc__tag(value):
    return f'{float(value):g}'.replace('.', 'p')


def finalize_4mu_final_mc__scopes():
    scopes = {'nominal_DD': dict(scope='nominal_DD', plane='DD', state='DD',
        mass_slice='nominal', mass_low=150., mass_high=None, nominal=True)}
    for mass, low, high, states in (('low', 0., 150., finalize_4mu_final_mc_STATES),
                                   ('intermediate', 150., 300., finalize_4mu_final_mc_STATES),
                                   ('all', 0., None, ('PP', 'PD', 'DP'))):
        for state in states:
            key = mass+'_'+state
            scopes[key] = dict(scope=key, plane=state, state=state, mass_slice=mass,
                              mass_low=low, mass_high=high, nominal=False)
    return scopes


def finalize_4mu_final_mc__load_mc(config, backgrounds, signals):
    scopes = finalize_4mu_final_mc__scopes()
    bplanes = {key: None for key in scopes}
    splane = {key: {} for key in scopes}
    flow_rows = []
    jobs = [(path, True) for path in backgrounds] + [(path, False) for path in signals]
    for index, (path, background) in enumerate(jobs, 1):
        loaded, payload = finalize_4mu_final_mc_common._comparison_payload(path, 'sixd')
        values, variances, edges = finalize_4mu_final_mc_common._comparison_current_cube(payload, path, config)
        # Reduce each displacement state once before forming the mass windows.
        reduced = {}
        for state in finalize_4mu_final_mc_STATES:
            first = finalize_4mu_final_mc_common._comparison_split(edges['disp0'], 2.5)
            second = finalize_4mu_final_mc_common._comparison_split(edges['disp1'], 2.5)
            ix = (slice(None), slice(finalize_4mu_final_mc_common._comparison_split(edges['iso1'], 0.),
                                    finalize_4mu_final_mc_common._comparison_split(edges['iso1'], .5)),
                  slice(first, None) if state[0] == 'P' else slice(None, first),
                  slice(second, None) if state[1] == 'P' else slice(None, second), slice(None), slice(None))
            reduced[state] = [a[ix].sum(axis=(1, 2, 3)) for a in (values, variances)]
        for scope, spec in scopes.items():
            low = finalize_4mu_final_mc_common._comparison_split(edges['ljlj_mass'], spec['mass_low'])
            high = None if spec['mass_high'] is None else finalize_4mu_final_mc_common._comparison_split(edges['ljlj_mass'], spec['mass_high'])
            state = spec['state']
            plane = finalize_4mu_final_mc_common._comparison_plane(*[a[:, :, low:high].sum(axis=2) for a in reduced[state]],
                                              edges['iso0'], edges['abs_dphi'])
            if background:
                bplanes[scope] = finalize_4mu_final_mc_common.add_candidate(bplanes[scope], plane, finalize_4mu_final_mc_common._comparison_group(path.stem))
            else:
                splane[scope][path.stem] = plane
        # Validate physical low+intermediate+high partitions before releasing the file.
        for state in finalize_4mu_final_mc_STATES:
            a, va = reduced[state]
            zero, split1, split2 = [finalize_4mu_final_mc_common._comparison_split(edges['ljlj_mass'], c) for c in (0., 150., 300.)]
            for array in (a, va):
                finalize_4mu_final_mc_np.testing.assert_allclose(array[:, :, zero:].sum(axis=2),
                    array[:, :, zero:split1].sum(axis=2)+array[:, :, split1:split2].sum(axis=2)+array[:, :, split2:].sum(axis=2),
                    rtol=1e-10, atol=1e-9)
            flow_rows.append(dict(sample=path.stem, state=state, background=background,
                physical_mass_sumw=float(a[1:, :, zero:].sum()),
                excluded_mass_underflow_sumw=float(a[1:, :, :zero].sum()),
                physical_mass_sumw2=float(va[1:, :, zero:].sum()),
                excluded_mass_underflow_sumw2=float(va[1:, :, :zero].sum()), passed=True))
        del loaded, payload, values, variances, reduced
        finalize_4mu_final_mc_gc.collect()
        if index % 12 == 0 or index == len(jobs):
            print(f'10 4mu MC projection: {index}/{len(jobs)}', flush=True)
    if any(len(x) != 120 for x in splane.values()):
        raise AssertionError('All scopes must contain 120 separate generated hypotheses.')
    return dict(backgrounds=bplanes, signals=splane, scopes=scopes, flow_audit=finalize_4mu_final_mc_pd.DataFrame(flow_rows))


def finalize_4mu_final_mc__parameters(sample):
    decay, mass, dark, lifetime = sample.split('_')
    return dict(signal_decay=decay, matching_decay=decay == '4Mu',
        m_B_GeV=float(mass.removesuffix('GeV').replace('p', '.')),
        m_ZD_GeV=float(dark.removesuffix('GeV').replace('p', '.')),
        ctau_mm=float(lifetime.removesuffix('mm').replace('p', '.')))


def finalize_4mu_final_mc__finite_summary(row, field, values):
    values = finalize_4mu_final_mc_np.asarray(values, float)
    good = values[finalize_4mu_final_mc_np.isfinite(values)]
    row[field+'_valid'] = len(good)
    row[field+'_median'] = float(finalize_4mu_final_mc_np.median(good)) if len(good) else finalize_4mu_final_mc_np.nan
    row[field+'_q10'] = float(finalize_4mu_final_mc_np.quantile(good, .1)) if len(good) else finalize_4mu_final_mc_np.nan
    row[field+'_q90'] = float(finalize_4mu_final_mc_np.quantile(good, .9)) if len(good) else finalize_4mu_final_mc_np.nan
    row[field+'_min'] = float(good.min()) if len(good) else finalize_4mu_final_mc_np.nan
    row[field+'_max'] = float(good.max()) if len(good) else finalize_4mu_final_mc_np.nan


def finalize_4mu_final_mc__gate_summary(rows, regions):
    valid = all(rows[r]['valid_background'] for r in regions)
    approved = valid and all(rows[r]['purity_pass'] for r in regions)
    if valid:
        worst = max(regions, key=lambda r: rows[r]['max_S_B'])
        maximum = rows[worst]['max_S_B']
        sample = rows[worst]['worst_sample']
    else:
        worst = next(r for r in regions if not rows[r]['valid_background'])
        maximum, sample = finalize_4mu_final_mc_np.nan, ''
    return dict(purity_pass=bool(approved), max_S_B=maximum, worst_region=worst,
                worst_sample=sample, positive_background=bool(valid))


def finalize_4mu_final_mc__scan_tables(projected, candidates):
    samples = sorted(projected['signals']['nominal_DD'])
    parameters = [finalize_4mu_final_mc__parameters(sample) for sample in samples]
    matching = finalize_4mu_final_mc_np.array([x['matching_decay'] for x in parameters])
    if len(samples) != 120 or matching.sum() != 60:
        raise AssertionError('Require 120 models including 60 matching 4Mu hypotheses.')
    geometry = candidates.to_dict('records')
    common_plane = projected['backgrounds']['nominal_DD']
    masks = finalize_4mu_final_mc_np.asarray([[finalize_4mu_final_mc_candidate_masks(common_plane, spec)[r] for r in finalize_4mu_final_mc_REGIONS] for spec in geometry], float)
    flat_masks = masks.reshape(len(geometry), 4, -1)
    scan_rows, region_rows, model_rows, component_rows = [], [], [], []
    for scope, bg in projected['backgrounds'].items():
        context = projected['scopes'][scope]
        signal_values = finalize_4mu_final_mc_np.asarray([projected['signals'][scope][s]['values'] for s in samples]).reshape(120, -1)
        signal_variances = finalize_4mu_final_mc_np.asarray([projected['signals'][scope][s]['variances'] for s in samples]).reshape(120, -1)
        all_s = finalize_4mu_final_mc_np.einsum('sf,crf->csr', signal_values, flat_masks, optimize=True)
        all_vs = finalize_4mu_final_mc_np.einsum('sf,crf->csr', signal_variances, flat_masks, optimize=True)
        all_b = finalize_4mu_final_mc_np.einsum('f,crf->cr', bg['values'].ravel(), flat_masks, optimize=True)
        all_vb = finalize_4mu_final_mc_np.einsum('f,crf->cr', bg['variances'].ravel(), flat_masks, optimize=True)
        ref_mask = finalize_4mu_final_mc_np.asarray(finalize_4mu_final_mc_candidate_masks(bg, dict(a_iso=.3, a_phi=2.6, b_iso=.3, c_phi=2.6))['A'], float).ravel()
        reference_s = signal_values @ ref_mask
        reference_b = float(bg['values'].ravel() @ ref_mask)
        reference_vb = float(bg['variances'].ravel() @ ref_mask)
        ref_z = finalize_4mu_final_mc_review.asimov_significance(reference_s, reference_b)
        ref_zv = finalize_4mu_final_mc_review.asimov_significance(reference_s, reference_b, reference_vb)
        for cindex, spec in enumerate(geometry):
            b, vb = all_b[cindex], all_vb[cindex]
            s, vs = all_s[cindex], all_vs[cindex]
            identity = dict(spec, **context, source='MC', topology='4mu')
            current_regions = {}
            for ridx, region in enumerate(finalize_4mu_final_mc_REGIONS):
                valid = bool(finalize_4mu_final_mc_np.isfinite([b[ridx], vb[ridx]]).all() and b[ridx] > 1e-8 and vb[ridx] >= 0)
                worst = int(finalize_4mu_final_mc_np.argmax(s[:, ridx]))
                signal, vsignal = float(s[worst, ridx]), float(vs[worst, ridx])
                maximum = signal/b[ridx] if valid else finalize_4mu_final_mc_np.nan
                ratio_error = finalize_4mu_final_mc_np.sqrt(vsignal/b[ridx]**2+signal**2*vb[ridx]/b[ridx]**4) if valid else finalize_4mu_final_mc_np.nan
                row = dict(identity, region=region, background=b[ridx], var_background=vb[ridx],
                    background_error=finalize_4mu_final_mc_np.sqrt(vb[ridx]), neff_background=b[ridx]**2/vb[ridx] if vb[ridx] > 0 else finalize_4mu_final_mc_np.nan,
                    max_S_B=maximum, max_S_B_error=ratio_error, worst_sample=samples[worst],
                    signal_max=signal, var_signal_max=vsignal, signal_error=finalize_4mu_final_mc_np.sqrt(vsignal),
                    neff_signal_max=signal**2/vsignal if vsignal > 0 else finalize_4mu_final_mc_np.nan,
                    valid_background=valid, signal_hypotheses=120,
                    purity_pass=bool(valid and finalize_4mu_final_mc_np.isfinite(maximum) and maximum < .1),
                    nominal_DATA_evaluated=False)
                current_regions[region] = row
                region_rows.append(row)
                for component, plane in bg.get('components', {}).items():
                    mask = masks[cindex, ridx]
                    component_rows.append(dict(identity, region=region, component=component,
                        **{'yield': float((plane['values']*mask).sum()),
                           'variance': float((plane['variances']*mask).sum())}))
            statistics = finalize_4mu_final_mc_statistics_from_regions(dict(zip(finalize_4mu_final_mc_REGIONS, b)), dict(zip(finalize_4mu_final_mc_REGIONS, vb)))
            row = dict(identity, **statistics, signal_hypotheses=120, matching_hypotheses=60)
            for subset, letters in (('BCD', 'BCD'), ('ABCD', 'ABCD')):
                gate = finalize_4mu_final_mc__gate_summary(current_regions, letters)
                row.update({'purity_pass_'+subset: gate['purity_pass'], 'max_S_B_'+subset: gate['max_S_B'],
                    'worst_'+subset+'_region': gate['worst_region'],
                    'worst_'+subset+'_sample': gate['worst_sample'],
                    'positive_'+subset: gate['positive_background']})
            row['purity_pass'] = row['purity_pass_BCD'] if scope == 'nominal_DD' else row['purity_pass_ABCD']
            row['gate_reason'] = ('CENTRAL_MC_PURITY_PASS_REQUIRES_DATA_AUDIT' if row['purity_pass'] else
                'NONPOSITIVE_OR_INVALID_MC_BACKGROUND' if not (row['positive_BCD'] if scope == 'nominal_DD' else row['positive_ABCD'])
                else 'CENTRAL_MC_CONTAMINATION_GE_10_PERCENT')
            row['prediction_relative_error'] = row['pred_error']/row['pred'] if row['pred'] > 0 else finalize_4mu_final_mc_np.nan
            z = finalize_4mu_final_mc_review.asimov_significance(s[:, 0], b[0])
            zv = finalize_4mu_final_mc_review.asimov_significance(s[:, 0], b[0], vb[0])
            ssb = finalize_4mu_final_mc_np.divide(s[:, 0], finalize_4mu_final_mc_np.sqrt(finalize_4mu_final_mc_np.maximum(s[:, 0]+b[0], 0)),
                out=finalize_4mu_final_mc_np.full(120, finalize_4mu_final_mc_np.nan), where=(s[:, 0]>=0)&(s[:, 0]+b[0]>0))
            sret = finalize_4mu_final_mc_np.divide(s[:, 0], reference_s, out=finalize_4mu_final_mc_np.full(120, finalize_4mu_final_mc_np.nan), where=reference_s > 0)
            zret = finalize_4mu_final_mc_np.divide(z, ref_z, out=finalize_4mu_final_mc_np.full(120, finalize_4mu_final_mc_np.nan), where=finalize_4mu_final_mc_np.isfinite(ref_z)&(ref_z>0))
            zvret = finalize_4mu_final_mc_np.divide(zv, ref_zv, out=finalize_4mu_final_mc_np.full(120, finalize_4mu_final_mc_np.nan), where=finalize_4mu_final_mc_np.isfinite(ref_zv)&(ref_zv>0))
            neff_s = finalize_4mu_final_mc_np.divide(s[:, 0]**2, vs[:, 0], out=finalize_4mu_final_mc_np.full(120, finalize_4mu_final_mc_np.nan), where=vs[:, 0]>0)
            for label, vector in [('signal_A', s[:, 0]), ('signal_A_retention_reference', sret),
                                  ('z_known_b', z), ('z_mc_stat_diagnostic', zv),
                                  ('s_sqrt_s_plus_b', ssb), ('z_known_b_ratio_reference', zret),
                                  ('z_mc_stat_ratio_reference', zvret), ('signal_neff_A', neff_s)]:
                finalize_4mu_final_mc__finite_summary(row, label, vector[matching])
            row['background_A_retention_reference'] = b[0]/reference_b if reference_b > 0 else finalize_4mu_final_mc_np.nan
            row['low_stat_signal_hypotheses'] = int(finalize_4mu_final_mc_np.count_nonzero(~finalize_4mu_final_mc_np.isfinite(neff_s[matching]) | (neff_s[matching] < 10)))
            scan_rows.append(row)
            for midx, (sample, param) in enumerate(zip(samples, parameters)):
                model = dict(identity, sample=sample, **param,
                    z_known_b=z[midx], z_mc_stat_diagnostic=zv[midx],
                    s_sqrt_s_plus_b=ssb[midx], signal_A_retention_reference=sret[midx],
                    z_known_b_ratio_reference=zret[midx], z_mc_stat_ratio_reference=zvret[midx],
                    reference_S_A=reference_s[midx], reference_B_A=reference_b,
                    reference_var_B_A=reference_vb, reference_z_known_b=ref_z[midx],
                    b=b[0], var_b=vb[0], neff_s=neff_s[midx],
                    nominal_DATA_evaluated=False)
                for ridx, region in enumerate(finalize_4mu_final_mc_REGIONS):
                    model.update({'S_'+region: s[midx, ridx], 'var_S_'+region: vs[midx, ridx],
                        'B_'+region: b[ridx], 'var_B_'+region: vb[ridx],
                        'S_B_'+region: s[midx, ridx]/b[ridx] if current_regions[region]['valid_background'] else finalize_4mu_final_mc_np.nan})
                model_rows.append(model)
    scan = finalize_4mu_final_mc_pd.DataFrame(scan_rows)
    regions = finalize_4mu_final_mc_pd.DataFrame(region_rows)
    return dict(mc_scan=scan, mc_regions=regions, mc_signals=finalize_4mu_final_mc_pd.DataFrame(model_rows),
        mc_background_components=finalize_4mu_final_mc_pd.DataFrame(component_rows),
        mc_flow_audit=projected['flow_audit'].copy(),
        mc_nominal_scan=scan.loc[scan.scope.eq('nominal_DD')].copy(),
        mc_vr_scan=scan.loc[~scan.scope.eq('nominal_DD')].copy())


def finalize_4mu_final_mc__regression(tables, projected, notebook_dir):
    checks = []
    directory = notebook_dir / '04_4mu_VR2_summary_outputs'
    old = finalize_4mu_final_mc_pd.read_csv(directory / 'sr_reference_grid.csv')
    old_signals = finalize_4mu_final_mc_pd.read_csv(directory / 'sr_reference_signal_points.csv')
    numeric = tables['mc_scan']
    models = tables['mc_signals']
    loose = numeric.loc[numeric.scope.eq('nominal_DD') & numeric.strategy.eq('loose')]
    fields = list(finalize_4mu_final_mc_REGIONS)+['var_'+r for r in finalize_4mu_final_mc_REGIONS]
    for row in loose.itertuples():
        prior = old.loc[finalize_4mu_final_mc_np.isclose(old.tx, row.a_iso)&finalize_4mu_final_mc_np.isclose(old.ty, row.a_phi)].iloc[0]
        finalize_4mu_final_mc_np.testing.assert_allclose([getattr(row, k) for k in fields], prior[fields].to_numpy(float),
                                   rtol=1e-9, atol=1e-9)
        current = models.loc[models.scope.eq('nominal_DD') & models.candidate_id.eq(row.candidate_id)].set_index('sample').sort_index()
        previous = old_signals.loc[finalize_4mu_final_mc_np.isclose(old_signals.tx, row.a_iso)&finalize_4mu_final_mc_np.isclose(old_signals.ty, row.a_phi)].set_index('sample').sort_index()
        signal_fields = ['S_'+r for r in finalize_4mu_final_mc_REGIONS]+['var_S_'+r for r in finalize_4mu_final_mc_REGIONS]
        assert len(current)==len(previous)==120
        finalize_4mu_final_mc_np.testing.assert_allclose(current[signal_fields].to_numpy(float), previous[signal_fields].to_numpy(float),
                                   rtol=1e-9, atol=1e-9)
        checks.append(dict(scope='nominal_DD', candidate_id=row.candidate_id,
            check='Original MC moments reproduce cached04 nominal grid and 120 signal models', passed=True))
    # Independent region lookup verifies buffered rectangles without reusing the direct mask path.
    for row in numeric.loc[numeric.scope.eq('nominal_DD') & ~numeric.strategy.eq('loose')].itertuples():
        coordinates = dict(A=(row.a_iso,row.a_phi),B=(row.b_iso,row.a_phi),
                           C=(row.a_iso,row.c_phi),D=(row.b_iso,row.c_phi))
        for r, (tx, ty) in coordinates.items():
            cached = old.loc[finalize_4mu_final_mc_np.isclose(old.tx,tx)&finalize_4mu_final_mc_np.isclose(old.ty,ty)].iloc[0]
            finalize_4mu_final_mc_np.testing.assert_allclose([getattr(row,r),getattr(row,'var_'+r)],
                [cached[r],cached['var_'+r]],rtol=1e-9,atol=1e-9)
        checks.append(dict(scope='nominal_DD', candidate_id=row.candidate_id,
            check='Buffered MC rectangles reproduce independent cached04 coordinate lookup', passed=True))
    old_vr = finalize_4mu_final_mc_pd.read_csv(directory / 'grid.csv')
    old_vr = old_vr.loc[old_vr.source.eq('MC')]
    for state in ('PP','PD','DP'):
        for row in numeric.loc[numeric.scope.eq('intermediate_'+state) & numeric.strategy.eq('loose')].itertuples():
            previous = old_vr.loc[old_vr.plane.eq(state)&finalize_4mu_final_mc_np.isclose(old_vr.tx,row.a_iso)&finalize_4mu_final_mc_np.isclose(old_vr.ty,row.a_phi)].iloc[0]
            finalize_4mu_final_mc_np.testing.assert_allclose([getattr(row,k) for k in fields], previous[fields].to_numpy(float),
                                       rtol=1e-8,atol=1e-8)
        checks.append(dict(scope='intermediate_'+state,candidate_id='all20loose',
            check='Intermediate PP/PD/DP MC reproduce cached04 region moments',passed=True))
    low_directory = notebook_dir / '02_4mu_VR1_summary_outputs'
    low_grid = finalize_4mu_final_mc_pd.read_csv(low_directory / 'grid.csv')
    low_grid = low_grid.loc[low_grid.source.eq('MC')]
    low_signals = finalize_4mu_final_mc_pd.read_csv(low_directory / 'signal_points.csv')
    intermediate_signals = finalize_4mu_final_mc_pd.read_csv(directory / 'signal_points.csv')
    signal_fields = ['S_'+r for r in finalize_4mu_final_mc_REGIONS]+['var_S_'+r for r in finalize_4mu_final_mc_REGIONS]
    for mass, states, prior_signals in [('low', finalize_4mu_final_mc_STATES, low_signals),
                                       ('intermediate', ('PP','PD','DP'), intermediate_signals)]:
        for state in states:
            rows = numeric.loc[numeric.scope.eq(mass+'_'+state) & numeric.strategy.eq('loose')]
            for row in rows.itertuples():
                if mass == 'low':
                    previous = low_grid.loc[low_grid.plane.eq(state) & finalize_4mu_final_mc_np.isclose(low_grid.tx,row.a_iso)
                                            & finalize_4mu_final_mc_np.isclose(low_grid.ty,row.a_phi)].iloc[0]
                    finalize_4mu_final_mc_np.testing.assert_allclose([getattr(row,k) for k in fields],previous[fields].to_numpy(float),
                                               rtol=1e-8,atol=1e-8)
                before = prior_signals.loc[prior_signals.plane.eq(state) & finalize_4mu_final_mc_np.isclose(prior_signals.tx,row.a_iso)
                    & finalize_4mu_final_mc_np.isclose(prior_signals.ty,row.a_phi)].set_index('sample').sort_index()
                current = models.loc[models.scope.eq(mass+'_'+state)
                    & models.candidate_id.eq(row.candidate_id)].set_index('sample').sort_index()
                assert len(current) == len(before) == 120
                finalize_4mu_final_mc_np.testing.assert_allclose(current[signal_fields].to_numpy(float),before[signal_fields].to_numpy(float),
                                           rtol=1e-8,atol=1e-8)
            checks.append(dict(scope=mass+'_'+state,candidate_id='all20loose',
                check='VR MC background and 120 signal moments reproduce cached02/04',passed=True))
    for anchor in ('WP2','WP1'):
        rows = numeric.loc[numeric.scope.eq('nominal_DD') & numeric.anchor.eq(anchor)]
        for field in ['A','var_A','z_known_b_median','z_mc_stat_diagnostic_median','s_sqrt_s_plus_b_median']:
            finalize_4mu_final_mc_np.testing.assert_allclose(rows[field], rows[field].iloc[0], rtol=1e-12,atol=1e-12,equal_nan=True)
        checks.append(dict(scope='nominal_DD',candidate_id=anchor,
            check='Fixed-A buffer exactly preserves A sumw/sumw2 and all sensitivity summaries',passed=True))
    assert len(loose)==20 and models.loc[models.scope.eq('nominal_DD')].groupby('candidate_id').size().eq(120).all()
    checks.append(dict(scope='all',candidate_id='all',
        check='120 individual models, 60 matching4Mu, and physical-mass partition checks retained',passed=True))
    checks.append(dict(scope='all',candidate_id='all',
        check='No DATA values or files read by MC computation',passed=True))
    return finalize_4mu_final_mc_pd.DataFrame(checks)


def finalize_4mu_final_mc__projection_signature():
    """Fingerprint raw projection only, so table/plot edits reuse cached planes."""
    tree = finalize_4mu_final_mc_ast.parse(finalize_4mu_final_mc_Path(__file__).read_text())
    names = {'finalize_4mu_final_mc__load_mc', 'finalize_4mu_final_mc__scopes', 'finalize_4mu_final_mc_project_cube'}
    code = [finalize_4mu_final_mc_ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, finalize_4mu_final_mc_ast.FunctionDef) and node.name in names]
    if len(code) != len(names):
        raise AssertionError('The consolidated projection signature must cover all three functions.')
    return finalize_4mu_final_mc_hashlib.sha256(finalize_4mu_final_mc_json.dumps(dict(functions=code, states=finalize_4mu_final_mc_STATES), sort_keys=True).encode()).hexdigest()


# ========================================================================
# abcd_comprehensive: original study API
# ========================================================================
"""Post-discussion diagnostics appended to the unchanged notebook-07 summary.

Nominal selections are MC only. No nominal DATA values are loaded, projected,
or inferred; the DATA sensitivity section records access decisions explicitly.
"""
from datetime import datetime as finalize_comprehensive_datetime; from datetime import timezone as finalize_comprehensive_timezone
from pathlib import Path as finalize_comprehensive_Path
import json as finalize_comprehensive_json

import numpy as finalize_comprehensive_np
import pandas as finalize_comprehensive_pd
from matplotlib.backends.backend_pdf import PdfPages as finalize_comprehensive_PdfPages

finalize_comprehensive_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_comprehensive_review = _FINALIZE_NAMESPACES['review_mc_sr_working_points']
finalize_comprehensive_run_mass_inclusive = _FINALIZE_NAMESPACES['abcd_mass_inclusive'].run_mass_inclusive


finalize_comprehensive_REFERENCE = (.30, 2.6)
finalize_comprehensive_ORDER = ('PP', 'PD', 'DP')


def finalize_comprehensive_comprehensive_points(focus):
    points = focus['tables']['working_points'][['wp', 'tx', 'ty']].copy()
    if points.duplicated(['tx', 'ty']).any() or len(points) != 5:
        raise ValueError('Keep the existing five notebook-07 working points.')
    reference = finalize_comprehensive_pd.DataFrame([dict(wp='Reference', tx=finalize_comprehensive_REFERENCE[0], ty=finalize_comprehensive_REFERENCE[1])])
    if any(finalize_comprehensive_np.isclose(points.tx, finalize_comprehensive_REFERENCE[0]) & finalize_comprehensive_np.isclose(points.ty, finalize_comprehensive_REFERENCE[1])):
        raise ValueError('This extension expects the current top five without a forced reference.')
    return finalize_comprehensive_pd.concat([reference, points], ignore_index=True)


def finalize_comprehensive__read(path, manifest, role):
    path = finalize_comprehensive_Path(path)
    manifest.append(finalize_comprehensive_common.candidate_input_record(path, role))
    return finalize_comprehensive_pd.read_csv(path, float_precision='round_trip')


def finalize_comprehensive__at(frame, tx, ty):
    return frame.loc[finalize_comprehensive_np.isclose(frame.tx, tx, rtol=0, atol=1e-12)
                     & finalize_comprehensive_np.isclose(frame.ty, ty, rtol=0, atol=1e-12)]


def finalize_comprehensive_contamination_diagnostics(focus, signal_grid, config):
    """Separate region envelopes, fixed-hypothesis yields and global maxima."""
    regions = focus['tables']['full_purity'].copy()
    grid = focus['tables']['full_closure'].query('source == "MC"')
    records, envelopes = [], []
    for plane in finalize_comprehensive_ORDER:
        background = grid.loc[grid.plane.eq(plane)]
        signals = signal_grid.loc[signal_grid.plane.eq(plane)]
        indexed = signals.set_index(['tx', 'ty', 'sample'])
        for tx in sorted({.2, .3}):
            reference_signals = finalize_comprehensive__at(signals, tx, finalize_comprehensive_REFERENCE[1]).set_index('sample')
            reference_background = finalize_comprehensive__at(background, tx, finalize_comprehensive_REFERENCE[1]).iloc[0]
            fixed = {}
            for region in 'ABCD':
                row = finalize_comprehensive__at(regions.loc[regions.plane.eq(plane) & regions.region.eq(region)],
                          tx, finalize_comprehensive_REFERENCE[1]).iloc[0]
                fixed[region] = row.worst_sample
            for ty in config['scan_abs_dphi']:
                b = finalize_comprehensive__at(background, tx, ty).iloc[0]
                region_rows = finalize_comprehensive__at(regions.loc[regions.plane.eq(plane)], tx, ty)
                for region in 'ABCD':
                    sample = fixed[region]
                    if finalize_comprehensive_pd.isna(sample) or sample not in reference_signals.index:
                        continue
                    s = indexed.loc[(tx, ty, sample)]
                    sr = reference_signals.loc[sample]
                    background_value, signal_value = float(b[region]), float(s['S_' + region])
                    envelope = region_rows.loc[region_rows.region.eq(region)].iloc[0]
                    records.append(dict(
                        plane=plane, tx=tx, ty=ty, region=region, sample=sample,
                        signal=signal_value, background=background_value,
                        var_signal=float(s['var_S_' + region]), var_background=float(b['var_' + region]),
                        neff_background=float(b['neff_' + region]),
                        fixed_S_B=signal_value / background_value if background_value > 1e-8 else finalize_comprehensive_np.nan,
                        max_S_B=envelope.max_S_B, worst_sample=envelope.worst_sample,
                        signal_retention=signal_value / sr['S_' + region] if sr['S_' + region] > 0 else finalize_comprehensive_np.nan,
                        background_retention=background_value / reference_background[region]
                        if reference_background[region] > 0 else finalize_comprehensive_np.nan))
                sample, region = b.worst_sample, b.worst_region
                valid = isinstance(sample, str) and sample and region in 'ABCD'
                s = indexed.loc[(tx, ty, sample)] if valid else None
                envelopes.append(dict(
                    plane=plane, tx=tx, ty=ty, worst_region=region, worst_sample=sample,
                    max_S_B=b.max_S_B, background=float(b[region]) if valid else finalize_comprehensive_np.nan,
                    var_background=float(b['var_' + region]) if valid else finalize_comprehensive_np.nan,
                    neff_background=float(b['neff_' + region]) if valid else finalize_comprehensive_np.nan,
                    signal=float(s['S_' + region]) if valid else finalize_comprehensive_np.nan,
                    var_signal=float(s['var_S_' + region]) if valid else finalize_comprehensive_np.nan))
    return dict(contamination_regions=regions, contamination_fixed=finalize_comprehensive_pd.DataFrame(records),
                contamination_envelope=finalize_comprehensive_pd.DataFrame(envelopes))


def finalize_comprehensive__geometry(plane, tx, ty):
    x = finalize_comprehensive_np.r_[-finalize_comprehensive_np.inf, plane['xedges']][:, None]
    y = finalize_comprehensive_np.r_[-finalize_comprehensive_np.inf, plane['yedges']][None, :]
    support = finalize_comprehensive_np.broadcast_to(x >= 0, plane['values'].shape)
    low_iso, high_angle = x < tx, y >= ty
    return dict(A=support & low_iso & high_angle, B=support & ~low_iso & high_angle,
                C=support & low_iso & ~high_angle, D=support & ~low_iso & ~high_angle)


def finalize_comprehensive_nominal_diagnostics(mass, points, background, signals, purity):
    """MC validation and explicit geometric protection before any DATA access."""
    plane = mass['sr_plane']
    reference = finalize_comprehensive__at(background, *finalize_comprehensive_REFERENCE).iloc[0]
    reference_signals = finalize_comprehensive__at(signals, *finalize_comprehensive_REFERENCE).set_index('sample')
    significance = finalize_comprehensive_review.signal_significance('2mu2e', background, signals)
    original_a = finalize_comprehensive__geometry(plane, *finalize_comprehensive_REFERENCE)['A']
    candidate_a = finalize_comprehensive_np.zeros_like(original_a)
    for point in points.loc[points.wp.ne('Reference')].itertuples():
        candidate_a |= finalize_comprehensive__geometry(plane, point.tx, point.ty)['A']
    protected = original_a | candidate_a
    summaries, components, audits, statuses, bias, validations = [], [], [], [], [], []
    metrics = list('ABCD') + ['var_' + region for region in 'ABCD'] + ['pred', 'pred_error', 'kappa', 'kappa_error']
    for point in points.itertuples():
        raw = finalize_comprehensive_common.plane_statistics(plane, point.tx, point.ty)
        cached = finalize_comprehensive__at(background, point.tx, point.ty).iloc[0]
        finalize_comprehensive_np.testing.assert_allclose([raw[k] for k in metrics], cached[metrics].to_numpy(float),
                                   rtol=1e-8, atol=1e-8, equal_nan=True)
        chosen = finalize_comprehensive__at(signals, point.tx, point.ty).set_index('sample')
        primary = chosen.loc[chosen.index.str.startswith('2Mu2E_')]
        ratio = primary.S_A / reference_signals.loc[primary.index, 'S_A'].replace(0, finalize_comprehensive_np.nan)
        z = finalize_comprehensive__at(significance.loc[significance.matching_decay], point.tx, point.ty)
        summaries.append(dict(wp=point.wp, **raw,
                              signal_retention_median=ratio.median(), signal_retention_q10=ratio.quantile(.1),
                              signal_retention_q90=ratio.quantile(.9),
                              background_retention=raw['A'] / reference.A,
                              signal_median=primary.S_A.median(), signal_q10=primary.S_A.quantile(.1),
                              signal_q90=primary.S_A.quantile(.9), signal_hypotheses=len(primary),
                              z_ratio_median=z.z_known_b_ratio.median(),
                              z_ratio_q10=z.z_known_b_ratio.quantile(.1), z_ratio_q90=z.z_known_b_ratio.quantile(.9)))
        for sample, sample_plane in mass['sr_sample_planes'].items():
            values = finalize_comprehensive_common._comparison_regions(sample_plane, point.tx, point.ty)
            for region, entry in values.items():
                components.append(dict(wp=point.wp, tx=point.tx, ty=point.ty, sample=sample,
                                       component=finalize_comprehensive_common._comparison_group(sample), region=region,
                                       **{'yield': entry['value']}, variance=entry['variance'],
                                       error=entry['error'], neff=entry['neff']))
        masks = finalize_comprehensive__geometry(plane, point.tx, point.ty)
        for region in 'BCD':
            pr = finalize_comprehensive__at(purity.loc[purity.region.eq(region)], point.tx, point.ty).iloc[0]
            valid = bool(finalize_comprehensive_np.isfinite(cached[region]) and cached[region] > 1e-8)
            signal_values = chosen['S_' + region]
            valid_signals = bool(len(chosen) == 120 and finalize_comprehensive_np.isfinite(signal_values).all() and signal_values.ge(0).all())
            passed = bool(valid and valid_signals and finalize_comprehensive_np.isfinite(pr.max_S_B) and pr.max_S_B < .1)
            old_overlap = masks[region] & original_a
            union_overlap = masks[region] & protected
            reason = []
            if not passed:
                reason.append('CONTAMINATION_OR_INVALID_MC_SUPPORT')
            if old_overlap.any():
                reason.append('OVERLAPS_ORIGINAL_BLINDED_A')
            if (union_overlap & ~original_a).any():
                reason.append('OVERLAPS_ADDITIONAL_CANDIDATE_A_PROTECTION')
            audits.append(dict(wp=point.wp, tx=point.tx, ty=point.ty, region=region,
                               max_S_B=pr.max_S_B, worst_sample=pr.worst_sample,
                               background=cached[region], background_error=cached[region + '_error'],
                               neff_background=cached['neff_' + region], purity_pass=passed,
                               overlap_original_A=bool(old_overlap.any()),
                               overlap_protected_A=bool(union_overlap.any()),
                               original_overlap_bins=int(old_overlap.sum()),
                               protected_overlap_bins=int(union_overlap.sum()),
                               access_allowed=passed and not union_overlap.any(),
                               nominal_data_evaluated=False,
                               reason='; '.join(reason) if reason else 'MC_AND_GEOMETRY_CHECKS_PASS_NO_DATA_READ'))
        audit = [entry for entry in audits if entry['wp'] == point.wp]
        allowed = all(entry['access_allowed'] for entry in audit)
        # This notebook has no nominal DATA loader. A future change of the blind
        # policy must be reviewed explicitly; never infer counts through VR data.
        reasons = sorted({entry['reason'] for entry in audit if not entry['access_allowed']})
        statuses.append(dict(wp=point.wp, tx=point.tx, ty=point.ty,
                             all_BCD_purity_pass=all(entry['purity_pass'] for entry in audit),
                             all_BCD_access_allowed=allowed, nominal_data_evaluated=False,
                             status='NOT_EVALUATED' if allowed else 'BLOCKED',
                             reason=' | '.join(reasons) if reasons else 'NO_NOMINAL_DATA_ACCESS_IN_THIS_NOTEBOOK',
                             data_B=finalize_comprehensive_np.nan, data_C=finalize_comprehensive_np.nan, data_D=finalize_comprehensive_np.nan,
                             predicted_background=finalize_comprehensive_np.nan, expected_z=finalize_comprehensive_np.nan))
        for sample, row in chosen.iterrows():
            rb, rc, rd = (row['S_B_' + region] for region in 'BCD')
            valid = bool(finalize_comprehensive_np.isfinite([rb, rc, rd]).all() and min(rb, rc, rd) >= 0)
            bias.append(dict(wp=point.wp, sample=sample, matching_decay=sample.startswith('2Mu2E_'),
                             r_B=rb, r_C=rc, r_D=rd,
                             relative_prediction_bias=(1 + rb) * (1 + rc) / (1 + rd) - 1 if valid else finalize_comprehensive_np.nan,
                             interpretation='Same-hypothesis signal injection into B/C/D; not a measured bias'))
            raw_signal = finalize_comprehensive_common._comparison_regions(mass['sr_signals'][sample], point.tx, point.ty)
            for region in 'ABCD':
                finalize_comprehensive_np.testing.assert_allclose([raw_signal[region]['value'], raw_signal[region]['variance']],
                                           [row['S_' + region], row['var_S_' + region]], rtol=1e-8, atol=1e-8)
        validations.append(dict(scope=point.wp, check='Nominal background and 120 signals reproduce the MC cache; no nominal DATA', passed=True))
    samples = finalize_comprehensive_pd.DataFrame(components)
    grouped = samples.groupby(['wp', 'tx', 'ty', 'component', 'region'], sort=False, as_index=False)[['yield', 'variance']].sum()
    grouped['error'] = finalize_comprehensive_np.sqrt(grouped.variance)
    for point in points.itertuples():
        for region in 'ABCD':
            part = samples.loc[samples.wp.eq(point.wp) & samples.region.eq(region)]
            row = finalize_comprehensive__at(background, point.tx, point.ty).iloc[0]
            finalize_comprehensive_np.testing.assert_allclose([part['yield'].sum(), part.variance.sum()],
                                       [row[region], row['var_' + region]], rtol=1e-8, atol=1e-8)
    bias_frame = finalize_comprehensive_pd.DataFrame(bias)
    bias_summary = []
    for wp, group in bias_frame.groupby('wp', sort=False):
        valid = group.loc[group.relative_prediction_bias.notna()]
        worst = valid.loc[valid.relative_prediction_bias.abs().idxmax()]
        bias_summary.append(dict(wp=wp, min_bias=valid.relative_prediction_bias.min(),
                                 max_bias=valid.relative_prediction_bias.max(),
                                 worst_sample=worst['sample'], worst_bias=worst.relative_prediction_bias))
    return dict(nominal_summary=finalize_comprehensive_pd.DataFrame(summaries), nominal_composition=grouped,
                nominal_samples=samples, nominal_bcd_audit=finalize_comprehensive_pd.DataFrame(audits),
                data_sensitivity_status=finalize_comprehensive_pd.DataFrame(statuses),
                nominal_signal_bias=bias_frame, nominal_signal_bias_summary=finalize_comprehensive_pd.DataFrame(bias_summary),
                nominal_validation=finalize_comprehensive_pd.DataFrame(validations))


def finalize_comprehensive_run_comprehensive(focus, config):
    if config['topology'] != '2mu2e' or config['selection_mode'] != 'sr_significance':
        raise ValueError('Keep the current MC-ranked 2mu2e notebook-07 summary.')
    points = finalize_comprehensive_comprehensive_points(focus)
    manifest = []
    directory = finalize_comprehensive_Path(config['vr2_output_dir'])
    signal_grid = finalize_comprehensive__read(directory / 'signal_points.csv', manifest, 'VR2 signal grid for S/B decomposition')
    background = finalize_comprehensive__read(directory / 'sr_reference_grid.csv', manifest, 'Nominal MC grid')
    signals = finalize_comprehensive__read(directory / 'sr_reference_signal_points.csv', manifest, 'Nominal MC signal grid')
    purity = finalize_comprehensive__read(directory / 'sr_reference_purity_grid_regions.csv', manifest, 'Nominal MC region contamination')
    tables = dict(working_points=points, **finalize_comprehensive_contamination_diagnostics(focus, signal_grid, config))
    mass = finalize_comprehensive_run_mass_inclusive(config, points)
    tables.update(finalize_comprehensive_nominal_diagnostics(mass, points, background, signals, purity))
    tables['validation'] = finalize_comprehensive_pd.concat([tables['nominal_validation'], mass['tables']['validation']], ignore_index=True)
    if not tables['validation'].passed.all():
        raise AssertionError('A comprehensive validation failed.')
    if tables['data_sensitivity_status'].nominal_data_evaluated.any():
        raise AssertionError('Nominal DATA must remain unopened.')
    tables['input_manifest'] = finalize_comprehensive_pd.DataFrame(manifest)
    return dict(focus=focus, mass=mass, tables=tables)


def finalize_comprehensive_export_comprehensive(result, config, figures, focused_exports, definitions):
    output = finalize_comprehensive_Path(config['notebook_dir']) / '07_2mu2e_comprehensive_outputs'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for order, (figure, name) in enumerate(figures, 1):
        figure.savefig(output / (name + '.png'), dpi=110, facecolor='white', bbox_inches='tight')
        records.append(dict(name=name, png=name + '.png', order=order))
    tables = {'focused_' + name: frame for name, frame in focused_exports.items()}
    tables.update(result['tables'])
    # Full signal grids are reproducible from production inputs; retain selected
    # contamination and fixed-hypothesis diagnostics rather than duplicating them.
    for name, frame in result['mass']['tables'].items():
        if isinstance(frame, finalize_comprehensive_pd.DataFrame) and name not in ('signal_points',):
            tables['mass_' + name] = frame
    tables['figure_manifest'] = finalize_comprehensive_pd.DataFrame(records)
    tables['figure_definitions'] = definitions
    canonical_exports = []
    for name, frame in tables.items():
        path = output / (name + '.csv')
        if name.startswith('focused_'):
            canonical = finalize_comprehensive_Path(config['output_dir']) / (name[len('focused_'):] + '.csv')
            csv_bytes = frame.to_csv(index=False).encode('utf-8')
            if (canonical.is_file() and canonical.read_bytes() == csv_bytes
                    and (not path.exists() or path.read_bytes() == csv_bytes)):
                canonical_exports.append(dict(alias=path.name,
                    canonical_path=str(canonical.resolve()), size_bytes=len(csv_bytes),
                    sha256=finalize_comprehensive_common._comparison_hash(canonical)))
                path.unlink(missing_ok=True)
                continue
        frame.to_csv(path, index=False)
    source_names = ('abcd_finalize_common.py', 'abcd_finalize_plots.py')
    source_records = [finalize_comprehensive_common.candidate_input_record(finalize_comprehensive_Path(config['notebook_dir']) / name, 'Comprehensive implementation')
                      for name in source_names]
    inputs = finalize_comprehensive_pd.concat([result['focus']['tables']['input_manifest'], result['tables']['input_manifest'],
                        result['mass']['tables']['input_manifest'], finalize_comprehensive_pd.DataFrame(source_records)], ignore_index=True)
    inputs.to_csv(output / 'input_manifest.csv', index=False)
    manifest = dict(created_utc=finalize_comprehensive_datetime.now(finalize_comprehensive_timezone.utc).isoformat(), configuration=config,
                    focused_methodology=result['focus']['metadata'],
                    mass_methodology=result['mass']['metadata'],
                    working_points=result['tables']['working_points'].to_dict('records'),
                    preserved_original_figures=8, figures=records,
                    nominal_data_evaluated=False,
                    nominal_protection='Original A at (0.30,2.6) is kept blind. The union of all five candidate A regions is additionally protected, and original-versus-additional overlaps are reported separately. Geometric overlaps count native cells, not DATA events.',
                    nominal_data_sensitivity='No nominal DATA loader is present. B/C/D contamination, original-A overlap and additional candidate-A protection are checked before any possible DATA use. Blocked expected sensitivities stay NaN; no hole-punched BC/D estimate is constructed.',
                    contamination='Maxima over 120 separate signal hypotheses and ABCD regions; denominator is direct background MC yield in that region, not BC/D. Fixed-model curves separate envelope switching from yield changes.',
                    signal_bias='Each (1+rB)(1+rC)/(1+rD)-1 uses the same signal hypothesis in all three sidebands.',
                    mass_selection='PP/PD/DP only, common partner isolation [0,0.5). All physical mass: [0,infinity), overflow retained, unphysical/invalid mass underflow excluded. Components [0,150), [150,300), [300,infinity) sum to the inclusive plane.',
                    mass_data_gate='All 120 hypotheses x ABCD max S/B_MC<0.10 with valid positive MC support; selected unapproved DATA values masked. DATA distributions also require the exact flow-folded per-bin purity mask.',
                    interpretation='PP/PD are main validation regions and DP is auxiliary. Low-mass domination can hide a higher-mass closure discrepancy; no inclusive-VR kappa correction is transferred to nominal DD.',
                    table_rows={name: len(frame) for name, frame in tables.items()},
                    implementation_sha256={finalize_comprehensive_Path(row['path']).name: row['sha256'] for row in source_records},
                    export_maintenance=dict(action='reuse_byte_identical_canonical_csv',
                        canonical_exports=canonical_exports, physics_values_changed=False,
                        exporter_sha256=finalize_comprehensive_common._comparison_hash(finalize_comprehensive_Path(__file__))))
    (output / 'manifest.json').write_text(finalize_comprehensive_json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + '\n')
    print('Comprehensive outputs:', output)
    return output


# ========================================================================
# abcd_final_check: original study API
# ========================================================================
"""MC diagnostics and previously approved VR caches for notebook 09.

The working points are fixed. No original DATA files, nominal DATA values,
event-level records, or new DATA selections are opened by this module.
"""
from pathlib import Path as finalize_final_check_Path
import gc as finalize_final_check_gc
import json as finalize_final_check_json

import numpy as finalize_final_check_np
import pandas as finalize_final_check_pd

finalize_final_check_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_final_check_build_vr2_focus_config = _FINALIZE_NAMESPACES['abcd_local_scan'].build_vr2_focus_config
finalize_final_check_project_mass_partitions = _FINALIZE_NAMESPACES['abcd_mass_inclusive'].project_mass_partitions


finalize_final_check_POINTS = (('Reference', .3, 2.6), ('WP1', .2, 2.6), ('WP2', .2, 2.7))
finalize_final_check_STATES = ('PP', 'PD', 'DP')
finalize_final_check_MASS_SLICES = ('low', 'intermediate', 'high', 'all')
finalize_final_check_NUMERIC_STATS = (list('ABCD') + ['var_' + r for r in 'ABCD']
                 + [r + '_error' for r in 'ABCD'] + ['neff_' + r for r in 'ABCD']
                 + ['pred', 'pred_error', 'kappa', 'kappa_error', 'pull',
                    'neff_min', 'min_neff', 'min_sideband_neff'])


def finalize_final_check_build_final_check_config(notebook_dir):
    config = finalize_final_check_build_vr2_focus_config(notebook_dir, selection_mode='sr_significance', force_reference=False)
    directory = finalize_final_check_Path(notebook_dir).resolve()
    config.update(notebook_path=directory / '09_MuIso_DPhi_2mu2e_WP1_WP2_FinalCheck.ipynb',
                  output_dir=directory / '09_2mu2e_final_check_outputs',
                  comprehensive_dir=directory / '07_2mu2e_comprehensive_outputs',
                  working_points=[dict(wp=wp, tx=tx, ty=ty) for wp, tx, ty in finalize_final_check_POINTS],
                  candidate_order=list(finalize_final_check_STATES), scan_iso_limits=(0., 2.),
                  scan_dphi_limits=(0., finalize_final_check_np.pi), fixed_candidates=True)
    return config


def finalize_final_check__read(path, manifest, role):
    path = finalize_final_check_Path(path)
    manifest.append(finalize_final_check_common.candidate_input_record(path, role))
    return finalize_final_check_pd.read_csv(path, float_precision='round_trip')


def finalize_final_check__selected(frame, points):
    if 'wp' in frame:
        return frame.loc[frame.wp.isin(points.wp)].copy()
    return points.merge(frame, on=['tx', 'ty'], validate='one_to_many')


def finalize_final_check__at(frame, tx, ty):
    return frame.loc[finalize_final_check_np.isclose(frame.tx, tx, rtol=0, atol=1e-12)
                     & finalize_final_check_np.isclose(frame.ty, ty, rtol=0, atol=1e-12)]


def finalize_final_check__metrics(value, variance):
    return {'yield': float(value), 'variance': float(variance), 'error': float(finalize_final_check_np.sqrt(variance)),
            'neff': float(value * value / variance) if variance > 0 else finalize_final_check_np.nan,
            'rel_error': float(finalize_final_check_np.sqrt(variance) / abs(value)) if value != 0 else finalize_final_check_np.nan}


def finalize_final_check__bounds(plane):
    return (finalize_final_check_np.r_[-finalize_final_check_np.inf, plane['xedges']], finalize_final_check_np.r_[plane['xedges'], finalize_final_check_np.inf],
            finalize_final_check_np.r_[-finalize_final_check_np.inf, plane['yedges']], finalize_final_check_np.r_[plane['yedges'], finalize_final_check_np.inf])


def finalize_final_check__masks(plane, tx, ty):
    xlow, _, ylow, _ = finalize_final_check__bounds(plane)
    x, y = xlow[:, None], ylow[None, :]
    valid_iso = finalize_final_check_np.broadcast_to(x >= 0, plane['values'].shape)
    low_iso, high_angle = x < tx, y >= ty
    return dict(A=valid_iso & low_iso & high_angle, B=valid_iso & ~low_iso & high_angle,
                C=valid_iso & low_iso & ~high_angle, D=valid_iso & ~low_iso & ~high_angle)


def finalize_final_check__load_backgrounds(config, manifest, validation):
    paths = sorted(finalize_final_check_Path(config['bkg_dir']).glob('*.coffea'))
    if len(paths) != 18:
        raise ValueError('The established production must contain 18 background MC samples.')
    nominal, samples = None, {}
    masses = {state: {mass: None for mass in finalize_final_check_MASS_SLICES} for state in finalize_final_check_STATES}
    mass_samples = {state: {mass: {} for mass in finalize_final_check_MASS_SLICES} for state in finalize_final_check_STATES}
    flow_rows = []
    for index, path in enumerate(paths, 1):
        loaded, payload = finalize_final_check_common._comparison_payload(path, 'sixd')
        weights, variances, edges = finalize_final_check_common._comparison_current_cube(payload, path, config)
        component = finalize_final_check_common._comparison_group(path.stem)
        plane = finalize_final_check_common.project_mc_sr(weights, variances, edges, config)
        samples[path.stem] = plane
        nominal = finalize_final_check_common.add_candidate(nominal, plane, component)
        for state in finalize_final_check_STATES:
            partitions, flow = finalize_final_check_project_mass_partitions(weights, variances, edges, state, config)
            flow_rows.append(dict(sample=path.stem, plane=state, **flow))
            for mass, plane in partitions.items():
                masses[state][mass] = finalize_final_check_common.add_candidate(masses[state][mass], plane, component)
                mass_samples[state][mass][path.stem] = plane
            validation.append(dict(scope=state, sample=path.stem,
                                   check='Raw MC physical mass slices conserve sumw and sumw2', passed=True))
        manifest.append(finalize_final_check_common.candidate_input_record(path, 'Original background MC only'))
        del loaded, payload, weights, variances
        finalize_final_check_gc.collect()
        if index % 6 == 0:
            print(f'09 nominal and mass-partition background MC loading: {index}/{len(paths)}', flush=True)
    return dict(nominal=nominal, nominal_samples=samples, mass=masses,
                mass_samples=mass_samples, mass_flow_audit=finalize_final_check_pd.DataFrame(flow_rows))


def finalize_final_check__nominal_tables(planes, points, cached, validation):
    nominal = planes['nominal']
    samples, regions, bins, support = [], [], [], []
    xlow, xhigh, ylow, yhigh = finalize_final_check__bounds(nominal)
    for point in points.itertuples():
        stats = finalize_final_check_common.plane_statistics(nominal, point.tx, point.ty)
        old = cached.loc[cached.wp.eq(point.wp)].iloc[0]
        finalize_final_check_np.testing.assert_allclose([stats[k] for k in finalize_final_check_NUMERIC_STATS], old[finalize_final_check_NUMERIC_STATS].to_numpy(float),
                                   rtol=1e-8, atol=1e-8, equal_nan=True)
        for sample, plane in planes['nominal_samples'].items():
            for region, row in finalize_final_check_common._comparison_regions(plane, point.tx, point.ty).items():
                samples.append(dict(wp=point.wp, tx=point.tx, ty=point.ty, sample=sample,
                                    component=finalize_final_check_common._comparison_group(sample), region=region,
                                    **finalize_final_check__metrics(row['value'], row['variance'])))
        for region, mask in finalize_final_check__masks(nominal, point.tx, point.ty).items():
            value, variance = nominal['values'][mask].sum(), nominal['variances'][mask].sum()
            finalize_final_check_np.testing.assert_allclose([value, variance], [stats[region], stats['var_' + region]],
                                       rtol=1e-10, atol=1e-8)
            regions.append(dict(wp=point.wp, tx=point.tx, ty=point.ty, region=region,
                                **finalize_final_check__metrics(value, variance)))
            for ix, iy in finalize_final_check_np.argwhere(mask):
                bins.append(dict(wp=point.wp, region=region, ix=int(ix), iy=int(iy),
                                 xlow=xlow[ix], xhigh=xhigh[ix], ylow=ylow[iy], yhigh=yhigh[iy],
                                 **finalize_final_check__metrics(nominal['values'][ix, iy], nominal['variances'][ix, iy])))
            part = finalize_final_check_pd.DataFrame([r for r in samples if r['wp'] == point.wp and r['region'] == region])
            top = part.loc[part['yield'].idxmax()]
            top_variance = part.loc[part.variance.idxmax()]
            indices = finalize_final_check_np.argwhere(mask)
            cell = int(finalize_final_check_np.argmax(nominal['values'][mask]))
            ix, iy = indices[cell]
            support.append(dict(wp=point.wp, region=region, top_sample=top['sample'],
                                top_sample_fraction=top['yield'] / value if value > 0 else finalize_final_check_np.nan,
                                top_sample_variance_fraction=top.variance / variance if variance > 0 else finalize_final_check_np.nan,
                                top_variance_sample=top_variance['sample'],
                                max_sample_variance_fraction=top_variance.variance / variance if variance > 0 else finalize_final_check_np.nan,
                                top_cell_ix=int(ix), top_cell_iy=int(iy),
                                top_cell_fraction=nominal['values'][ix, iy] / value if value > 0 else finalize_final_check_np.nan,
                                top_cell_variance_fraction=nominal['variances'][ix, iy] / variance if variance > 0 else finalize_final_check_np.nan,
                                max_cell_variance_fraction=nominal['variances'][mask].max() / variance if variance > 0 else finalize_final_check_np.nan,
                                occupied_native_bins=int(finalize_final_check_np.count_nonzero(nominal['variances'][mask] > 0))))
            finalize_final_check_np.testing.assert_allclose([part['yield'].sum(), part.variance.sum()],
                                       [value, variance], rtol=1e-10, atol=1e-8)
        validation.append(dict(scope=point.wp, check='Nominal MC samples, native cells and cached 07 ABCD statistics agree', passed=True))
    return dict(nominal_regions=finalize_final_check_pd.DataFrame(regions), nominal_samples=finalize_final_check_pd.DataFrame(samples),
                nominal_support=finalize_final_check_pd.DataFrame(support), nominal_bins=finalize_final_check_pd.DataFrame(bins), nominal_summary=cached)


def finalize_final_check__migration_tables(planes, validation):
    nominal = planes['nominal']
    a0, a1, a2 = (finalize_final_check__masks(nominal, tx, ty)['A'] for _, tx, ty in finalize_final_check_POINTS)
    if (a1 & ~a0).any() or (a2 & ~a1).any():
        raise AssertionError('Expected strictly nested nominal MC A selections.')
    masks = dict(reference_A=a0, iso_removed=a0 & ~a1, angle_removed=a1 & ~a2, kept_WP2=a2)
    labels = dict(reference_A='Reference', iso_removed='WP1', angle_removed='WP2', kept_WP2='WP2')
    rows = []
    for sample, plane in planes['nominal_samples'].items():
        for step, mask in masks.items():
            rows.append(dict(wp=labels[step], step=step, sample=sample,
                             component=finalize_final_check_common._comparison_group(sample),
                             **finalize_final_check__metrics(plane['values'][mask].sum(), plane['variances'][mask].sum())))
        for field in ('values', 'variances'):
            reference = plane[field][a0].sum()
            parts = sum(plane[field][masks[s]].sum() for s in ('iso_removed', 'angle_removed', 'kept_WP2'))
            finalize_final_check_np.testing.assert_allclose(reference, parts, rtol=1e-10, atol=1e-8)
        validation.append(dict(scope=sample, check='Disjoint iso/angle removals and kept WP2 partition reference A sumw and sumw2', passed=True))
    samples = finalize_final_check_pd.DataFrame(rows)
    grouped = samples.groupby(['wp', 'step'], sort=False)[['yield', 'variance']].sum().reset_index()
    # Itertuples renames the Python keyword "yield"; aggregate explicitly instead.
    summary = finalize_final_check_pd.DataFrame([dict(wp=row['wp'], step=row['step'], **finalize_final_check__metrics(row['yield'], row['variance']))
                            for _, row in grouped.iterrows()])
    reference = summary.loc[summary.step.eq('reference_A')].iloc[0]
    summary['reference_fraction'] = summary['yield'] / reference['yield']
    summary['variance_fraction'] = summary.variance / reference.variance
    return dict(migration_samples=samples, migration_summary=summary)


def finalize_final_check__cache_tables(config, points, manifest, validation):
    directory = finalize_final_check_Path(config['comprehensive_dir'])
    names = ('nominal_summary', 'nominal_bcd_audit', 'data_sensitivity_status',
             'nominal_signal_bias_summary', 'nominal_signal_bias', 'mass_mass_components', 'mass_selected_purity')
    caches = {name: finalize_final_check__selected(finalize_final_check__read(directory / (name + '.csv'), manifest, 'Notebook 07 ' + name), points)
              for name in names}
    grids = {}
    for mass, folder in [('low', finalize_final_check_Path(config['vr1_output_dir'])), ('intermediate', finalize_final_check_Path(config['vr2_output_dir']))]:
        docpath = folder / 'manifest.json'
        document = finalize_final_check_json.loads(docpath.read_text())
        settings = document['settings']
        expected_bounds = (0., 150.) if mass == 'low' else (150., 300.)
        if (settings['topology'] != '2mu2e' or settings['purity_limit'] != .1
                or (settings['mass_min'], settings['mass_max']) != expected_bounds
                or (settings['partner_iso_low'], settings['partner_iso_high']) != (0., .5)):
            raise ValueError('VR cache selection differs from the fixed final-check convention.')
        for state in finalize_final_check_STATES:
            spec = document['candidate_specs'][state]
            if spec['state0'] + spec['state1'] != state:
                raise ValueError('Cached VR state ordering differs.')
        manifest.append(finalize_final_check_common.candidate_input_record(docpath, mass + ' VR cache selection'))
        grid = finalize_final_check__read(folder / 'grid.csv', manifest, mass + ' VR cached approved DATA and MC')
        grids[mass] = finalize_final_check__selected(grid.loc[grid.plane.isin(finalize_final_check_STATES)], points)
    grids['all'] = finalize_final_check__selected(finalize_final_check__read(directory / 'mass_grid.csv', manifest, 'Notebook 07 all-mass gated grid'), points)
    docpath = directory / 'manifest.json'
    doc = finalize_final_check_json.loads(docpath.read_text())
    if doc['nominal_data_evaluated'] or doc['mass_methodology']['nominal_data_loaded']:
        raise ValueError('The source notebook must preserve nominal DATA blinding.')
    manifest.append(finalize_final_check_common.candidate_input_record(docpath, 'Notebook 07 mass and blind-mask provenance'))
    for mass, frame in grids.items():
        if len(frame) != 18 or not frame.plane.isin(finalize_final_check_STATES).all():
            raise AssertionError('Each cached mass selection requires 3 WPs x 3 states x 2 sources.')
        blocked = frame.loc[frame.source.eq('DATA') & ~frame.approved]
        if not blocked[finalize_final_check_NUMERIC_STATS].isna().all().all():
            raise AssertionError('A cached blocked DATA statistic is exposed.')
        if not frame.signal_hypotheses.eq(120).all():
            raise AssertionError('Every cached gate must include all 120 separate hypotheses.')
        validation.append(dict(scope=mass, check='Cached VR selection, 120-model gate and blinded DATA fields validated', passed=True))
    if caches['data_sensitivity_status'].nominal_data_evaluated.any():
        raise AssertionError('Nominal DATA must remain unevaluated.')
    return caches, grids


def finalize_final_check__mass_tables(planes, points, caches, grids, validation):
    regional = caches['mass_mass_components'].rename(columns={'background': 'yield', 'var_background': 'variance'}).copy()
    regional = regional.loc[regional.plane.isin(finalize_final_check_STATES)]
    inclusive = caches['mass_selected_purity'].copy()
    inclusive['mass_slice'] = 'all'
    purity = finalize_final_check_pd.concat([regional, inclusive], ignore_index=True)
    rows, compositions, mixing = [], [], []
    for point in points.itertuples():
        for state in finalize_final_check_STATES:
            state_stats = {}
            for mass in finalize_final_check_MASS_SLICES:
                actual = finalize_final_check_common.plane_statistics(planes['mass'][state][mass], point.tx, point.ty)
                pr = purity.loc[purity.wp.eq(point.wp) & purity.plane.eq(state) & purity.mass_slice.eq(mass)]
                if len(pr) != 4 or set(pr.region) != set('ABCD'):
                    raise AssertionError('Missing ABCD contamination entries for the selected mass slice.')
                valid = bool(finalize_final_check_np.isfinite([actual[r] for r in 'ABCD']).all() and min(actual[r] for r in 'ABCD') > 1e-8)
                finite_ratios = finalize_final_check_np.isfinite(pr.max_S_B).all()
                approved = bool(valid and finite_ratios and pr.approved.all() and pr.max_S_B.lt(.1).all())
                worst = pr.loc[pr.max_S_B.idxmax()] if finite_ratios else None
                gate = dict(approved=approved, purity_pass=approved, valid_background=valid,
                            max_S_B=float(worst.max_S_B) if worst is not None else finalize_final_check_np.nan,
                            worst_region=worst.region if worst is not None else '',
                            worst_sample=worst.worst_sample if worst is not None else '', signal_hypotheses=120,
                            gate_reason='' if approved else 'S/B >= 0.10 or invalid/nonpositive MC support')
                base = dict(wp=point.wp, plane=state, mass_slice=mass)
                rows.append(dict(**base, source='MC', **actual, **gate,
                                 data_evaluated=False, display_allowed=True))
                state_stats[mass] = actual
                if mass in grids:
                    cached = grids[mass].loc[grids[mass].wp.eq(point.wp) & grids[mass].plane.eq(state)]
                    mc = cached.loc[cached.source.eq('MC')].iloc[0]
                    finalize_final_check_np.testing.assert_allclose([actual[k] for k in finalize_final_check_NUMERIC_STATS], mc[finalize_final_check_NUMERIC_STATS].to_numpy(float),
                                               rtol=1e-8, atol=1e-8, equal_nan=True)
                    if bool(mc.approved) != approved:
                        raise AssertionError('Reconstructed mass gate differs from approved source cache.')
                    data = cached.loc[cached.source.eq('DATA')].iloc[0].to_dict()
                    data.update(base, data_evaluated=approved, display_allowed=approved)
                    # Do not expose the three complementary DATA totals together:
                    # all-low-intermediate would reconstruct the blinded high mass.
                    # No such DATA subtraction is calculated, even internally.
                    if point.wp == 'Reference' and mass == 'all' and approved:
                        for field in finalize_final_check_NUMERIC_STATS:
                            data[field] = finalize_final_check_np.nan
                        data.update(display_allowed=False, data_evaluated=False,
                                    gate_reason='WITHHELD_TO_PREVENT_BLINDED_HIGH_MASS_SUBTRACTION',
                                    status='WITHHELD_COMPLEMENTARY_MASS_PROTECTION',
                                    kappa_status='WITHHELD_COMPLEMENTARY_MASS_PROTECTION', point_pass=False)
                    rows.append(data)
                else:
                    if approved:
                        raise AssertionError('High-mass purity changed: no high-mass DATA cache is authorized here.')
                    blank = finalize_final_check_common._extension_blank_statistics(point.tx, point.ty)
                    rows.append(dict(**base, source='DATA', **blank, **gate,
                                     data_evaluated=False, display_allowed=False))
                for sample, plane in planes['mass_samples'][state][mass].items():
                    for region, entry in finalize_final_check_common._comparison_regions(plane, point.tx, point.ty).items():
                        compositions.append(dict(**base, sample=sample, component=finalize_final_check_common._comparison_group(sample),
                                                 region=region, **finalize_final_check__metrics(entry['value'], entry['variance'])))
                for region in 'ABCD':
                    pr_row = pr.loc[pr.region.eq(region)].iloc[0]
                    if mass != 'all':
                        finalize_final_check_np.testing.assert_allclose([actual[region], actual['var_' + region]],
                                                   [pr_row['yield'], pr_row.variance], rtol=1e-8, atol=1e-8)
                validation.append(dict(scope=f'{point.wp}/{state}/{mass}',
                                       check='Raw MC ABCD and purity agree with stored mass diagnostics; only approved cached DATA reused', passed=True))
            disjoint = [state_stats[mass] for mass in finalize_final_check_MASS_SLICES[:3]]
            full = state_stats['all']
            for field in list('ABCD') + ['var_' + r for r in 'ABCD']:
                finalize_final_check_np.testing.assert_allclose(sum(item[field] for item in disjoint), full[field], rtol=1e-10, atol=1e-8)
            supported = all(finalize_final_check_np.isfinite(item['pred']) and item['D'] > 0 for item in disjoint)
            sum_predictions = sum(item['pred'] for item in disjoint) if supported else finalize_final_check_np.nan
            difference = full['pred'] - sum_predictions if supported else finalize_final_check_np.nan
            b_weighted = sum_predictions / full['B'] if supported and full['B'] > 0 else finalize_final_check_np.nan
            d_weighted = full['C'] / full['D'] if full['D'] > 0 else finalize_final_check_np.nan
            mixing.append(dict(wp=point.wp, plane=state, pooled_prediction=full['pred'],
                               sum_slice_predictions=sum_predictions, mixing_difference=difference,
                               mixing_fraction=difference / full['pred'] if supported and full['pred'] > 0 else finalize_final_check_np.nan,
                               slices_supported=supported, weighted_C_over_D=b_weighted,
                               D_weighted_C_over_D=d_weighted,
                               interpretation='Central-value mass-mixture diagnostic; pooled and slice estimates are correlated'))
    closure = finalize_final_check_pd.DataFrame(rows)
    blocked = closure.loc[closure.source.eq('DATA') & ~closure.display_allowed]
    if not blocked[finalize_final_check_NUMERIC_STATS].isna().all().all():
        raise AssertionError('Blocked mass DATA must stay NaN.')
    composition = finalize_final_check_pd.DataFrame(compositions)
    for row in closure.loc[closure.source.eq('MC')].itertuples():
        selected = composition.loc[composition.wp.eq(row.wp) & composition.plane.eq(row.plane)
                                   & composition.mass_slice.eq(row.mass_slice)]
        for region in 'ABCD':
            part = selected.loc[selected.region.eq(region)]
            finalize_final_check_np.testing.assert_allclose([part['yield'].sum(), part.variance.sum()],
                                       [getattr(row, region), getattr(row, 'var_' + region)], rtol=1e-10, atol=1e-8)
    validation.append(dict(scope='Mass composition', check='All sample sums and mass partitions conserve ABCD sumw/sumw2', passed=True))
    validation.append(dict(scope='Complementary DATA protection',
                           check='No hidden DATA statistics exported; reference inclusive DATA withheld to prevent high-mass subtraction', passed=True))
    return dict(mass_closure=closure, mass_composition=composition, mass_mixing=finalize_final_check_pd.DataFrame(mixing),
                mass_purity_regions=purity, mass_flow_audit=planes['mass_flow_audit'])


def finalize_final_check_run_final_check(config):
    if (config['topology'] != '2mu2e' or config['purity_limit'] != .1
            or (config['partner_iso_low'], config['partner_iso_high']) != (0., .5)):
        raise ValueError('Keep 2mu2e, partner isolation [0,0.5), and the established 10% gate.')
    points = finalize_final_check_pd.DataFrame(config['working_points'])
    if list(points.itertuples(index=False, name=None)) != list(finalize_final_check_POINTS):
        raise ValueError('Notebook 09 fixes Reference, WP1 and WP2; no ranking or boundary scan is performed.')
    manifest, validation = [], []
    caches, grids = finalize_final_check__cache_tables(config, points, manifest, validation)
    planes = finalize_final_check__load_backgrounds(config, manifest, validation)
    tables = dict(working_points=points,
                  **finalize_final_check__nominal_tables(planes, points, caches['nominal_summary'], validation),
                  **finalize_final_check__migration_tables(planes, validation),
                  **finalize_final_check__mass_tables(planes, points, caches, grids, validation))
    for name in ('nominal_bcd_audit', 'data_sensitivity_status', 'nominal_signal_bias_summary', 'nominal_signal_bias'):
        tables[name] = caches[name]
    validation.append(dict(scope='Blinding', check='No original DATA loading or nominal DATA calculation; inherited five-candidate A protection retained', passed=True))
    tables['validation'] = finalize_final_check_pd.DataFrame(validation)
    manifest.append(finalize_final_check_common.candidate_input_record(finalize_final_check_Path(__file__), 'Final-check MC/cache implementation'))
    tables['input_manifest'] = finalize_final_check_pd.DataFrame(manifest)
    metadata = dict(
        original_background_files_opened=18, original_signal_files_opened=0, original_data_files_opened=0,
        nominal_data_evaluated=False, new_data_selection_evaluated=False,
        candidate_policy='Reference, WP1, WP2 fixed before this notebook; no DATA-driven ranking.',
        nominal_selection='DD, MJJ>=150 including overflow; partner isolation [0,0.5).',
        mass_selection='PP/PD/DP only. Low [0,150), intermediate [150,300), high [300,infinity); overflow included, mass underflow excluded.',
        signal_policy='Contamination caches retain maxima over 120 separate hypotheses; alternative hypotheses are never summed.',
        data_policy='Only pre-approved low/intermediate/all-mass cached VR statistics may be reused. High-mass DATA and every nominal DATA yield remain unexamined.',
        complementary_data_protection='Reference PP/PD all-mass DATA pass the original purity gate but their complete numeric statistics are additionally withheld in notebook 09. Exporting them together with low and intermediate DATA would permit reconstruction of the blinded high-mass complement. approved retains the original MC gate; display_allowed is the final release flag. No DATA complement is calculated.',
        blindness='Original reference A and the complete five-candidate A union from notebook 07 stay protected; focusing on two candidates does not relax the mask.',
        uncertainty='Disjoint histogram regions use sumw2; nested working-point differences use explicit disjoint masks, not independent-error subtraction.',
        event_information='Weighted histograms do not retain event IDs, actual raw event counts or individual event weights. Occupied bins and N_eff are support diagnostics, not event counts.',
        mixture='BC/D after pooling mass bins differs from sum of the component BC/D predictions. The difference is a correlated central-value diagnostic, not an independent uncertainty or a nominal correction.')
    return dict(config=config, tables=tables, planes=planes, metadata=metadata)


# ========================================================================
# abcd_sideband_validation: original study API
# ========================================================================
"""Cached VR validation for relaxed and buffered 2mu2e ABCD definitions.

No original DATA files are opened. New cached DATA measurements are evaluated
only after their integrated gate, source-cache gate and disclosure audit pass.
"""
from pathlib import Path as finalize_sideband_validation_Path
import json as finalize_sideband_validation_json

import numpy as finalize_sideband_validation_np
import pandas as finalize_sideband_validation_pd
from threadpoolctl import threadpool_limits as finalize_sideband_validation_threadpool_limits

finalize_sideband_validation_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_sideband_validation_candidate_coordinates = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].candidate_coordinates; finalize_sideband_validation_statistics_from_regions = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].statistics_from_regions


finalize_sideband_validation_REGIONS = tuple('ABCD')
finalize_sideband_validation_MEASUREMENT_FIELDS = (list(finalize_sideband_validation_REGIONS) + ['var_' + r for r in finalize_sideband_validation_REGIONS]
                      + [r + '_error' for r in finalize_sideband_validation_REGIONS] + ['neff_' + r for r in finalize_sideband_validation_REGIONS]
                      + ['pred', 'pred_error', 'kappa', 'kappa_error', 'pull',
                         'min_sideband_neff', 'min_neff', 'neff_min'])
finalize_sideband_validation_SCOPE_DEFINITIONS = [('VR1_DD', 'low', 'DD'), ('VR1_PP', 'low', 'PP'),
                     ('VR1_PD', 'low', 'PD'), ('VR1_DP', 'low', 'DP'),
                     ('VR2_PP', 'intermediate', 'PP'), ('VR2_PD', 'intermediate', 'PD'),
                     ('VR2_DP', 'intermediate', 'DP')]


def finalize_sideband_validation_run_sideband_validation(config, nominal_result, existing_tables, load_data=False, strata_mc=None):
    """Validate the MC-selected shortlist in seven explicit existing VRs.

    ``load_data=False`` assembles all gates and disclosure decisions first.
    With ``True``, only approved cached measurements are filled; no ROOT or
    original DATA coffea reader is used by this module.
    """
    if config['topology'] != '2mu2e' or config['purity_limit'] != .1:
        raise ValueError('This validation requires 2mu2e and strict S/B < 10%.')
    selection = nominal_result['tables']['sideband_selection'].copy()
    if selection.candidate_id.duplicated().any():
        raise AssertionError('The nominal MC shortlist must have unique candidate IDs.')
    inputs, validation = [], []
    caches = finalize_sideband_validation__load_caches(config, inputs, validation)
    strata_mc = finalize_sideband_validation__validated_strata_mc(config, existing_tables, strata_mc, inputs, validation)
    closure, purity, signal_rows, release_rows, complement_rows = [], [], [], [], []
    for scope, mass, state in finalize_sideband_validation_SCOPE_DEFINITIONS:
        cache = caches[mass]
        background = cache['grid'].loc[cache['grid'].source.eq('MC') & cache['grid'].plane.eq(state)]
        signals = cache['signals'].loc[cache['signals'].plane.eq(state)]
        data = cache['grid'].loc[cache['grid'].source.eq('DATA') & cache['grid'].plane.eq(state)]
        b_index = finalize_sideband_validation__background_index(background)
        s_index, names = finalize_sideband_validation__signal_index(signals)
        d_index = finalize_sideband_validation__background_index(data)
        x_edges = sorted(set([0., .2, .3, finalize_sideband_validation_np.inf] + selection.a_iso.tolist() + selection.b_iso.tolist()))
        y_edges = sorted(set([-finalize_sideband_validation_np.inf, 2.6, 2.7, finalize_sideband_validation_np.inf] + selection.a_phi.tolist() + selection.c_phi.tolist()))
        atomic = finalize_sideband_validation_cached_atomic_measurements(background, signals, x_edges, y_edges)
        prior, fine_prior, leaf_count = finalize_sideband_validation__existing_observations(existing_tables, mass, state, atomic['atoms'])
        fine_atomic = finalize_sideband_validation__fine_atomic_measurements(strata_mc, mass, state, atomic, validation) if leaf_count > 1 else None
        published = dict(prior)
        published_fine = dict(fine_prior)
        # MC-only ordering is fixed before any new cached DATA value is read.
        ordered = finalize_sideband_validation__release_order(selection)
        for spec in ordered.to_dict('records'):
            identity = {key: spec[key] for key in ('candidate_id', 'strategy', 'a_iso', 'a_phi', 'b_iso', 'c_phi', 'display_label')}
            identity.update(scope=scope, plane=state, mass_slice=mass)
            coordinates = finalize_sideband_validation_candidate_coordinates(spec)
            values, variances, signal_values, signal_variances = {}, {}, {}, {}
            source_pass = True
            source_fail = []
            for region, coordinate in coordinates.items():
                key = finalize_sideband_validation__key(*coordinate)
                b = b_index[key]
                sig = s_index[key]
                values[region], variances[region] = float(b[region]), float(b['var_' + region])
                signal_values[region] = sig['S_' + region].to_numpy(float)
                signal_variances[region] = sig['var_S_' + region].to_numpy(float)
                cached_source = d_index[key]
                if not bool(cached_source['approved']):
                    source_pass = False
                    source_fail.append(region)
            statistics = finalize_sideband_validation_statistics_from_regions(values, variances, config.get('min_neff', 10.))
            gate_rows = []
            for region in finalize_sideband_validation_REGIONS:
                ratios = finalize_sideband_validation__ratios(signal_values[region], values[region])
                good = bool(values[region] > 1e-8 and finalize_sideband_validation_np.isfinite(ratios).all()
                            and (signal_values[region] >= 0).all())
                index = int(finalize_sideband_validation_np.argmax(signal_values[region]))
                maximum = float(ratios[index]) if good else finalize_sideband_validation_np.nan
                gate_rows.append(dict(identity, region=region, background=values[region],
                                      var_background=variances[region], max_S_B=maximum,
                                      worst_sample=names[index], signal_hypotheses=len(names),
                                      valid_background=bool(values[region] > 1e-8),
                                      approved=bool(good and maximum < .1)))
            purity.extend(gate_rows)
            own_gate = all(row['approved'] for row in gate_rows)
            maximum = max(row['max_S_B'] for row in gate_rows) if all(finalize_sideband_validation_np.isfinite(row['max_S_B']) for row in gate_rows) else finalize_sideband_validation_np.nan
            masks = finalize_sideband_validation_candidate_masks(spec, atomic['atoms'])
            for region, mask in masks.items():
                finalize_sideband_validation_np.testing.assert_allclose([atomic['values'][mask].sum(), atomic['variances'][mask].sum()],
                    [values[region], variances[region]], rtol=1e-8, atol=1e-8)
                finalize_sideband_validation_np.testing.assert_allclose(atomic['signals'][:, mask].sum(axis=1), signal_values[region],
                                           rtol=1e-8, atol=1e-8)
                finalize_sideband_validation_np.testing.assert_allclose(atomic['signal_variances'][:, mask].sum(axis=1), signal_variances[region],
                                           rtol=1e-8, atol=1e-8)
            proposed = {spec['candidate_id'] + ':' + r: masks[r] for r in finalize_sideband_validation_REGIONS}
            audit = finalize_sideband_validation_disclosure_audit(published, proposed, atomic)
            fine_audit = finalize_sideband_validation_disclosure_audit(published_fine,
                {name: finalize_sideband_validation_np.tile(mask, leaf_count) for name, mask in proposed.items()}, fine_atomic) if fine_atomic else audit.iloc[:0].copy()
            fine_ok = bool(fine_audit.approved.all())
            fine_count = len(fine_audit)
            if len(fine_audit):
                fine_audit['audit_kind'] = 'FINE_' + fine_audit.audit_kind
                audit = finalize_sideband_validation_pd.concat([audit, fine_audit], ignore_index=True)
            if len(audit):
                for record in audit.to_dict('records'):
                    complement_rows.append(dict(identity, **record))
            complementary_pass = bool(audit.approved.all() if len(audit) else True) and fine_ok
            eligible = bool(own_gate and source_pass and complementary_pass)
            reason = ('PURITY_SOURCE_AND_DISCLOSURE_PASS' if eligible else
                      'VR_CONTAMINATION_OR_INVALID_BACKGROUND' if not own_gate else
                      'SOURCE_CACHE_BLOCKED' if not source_pass else
                      'UNSAFE_FINE_PARTITION_COMPLEMENT' if not fine_ok else
                      'UNSAFE_OR_UNSUPPORTED_RECONSTRUCTABLE_COMPLEMENT')
            release_rows.append(dict(identity, approved=own_gate, max_S_B=maximum,
                                     source_cache_approved=source_pass,
                                     source_cache_blocked_regions=','.join(source_fail),
                                     complementary_pass=complementary_pass,
                                     new_complement_checks=len(audit),
                                     failed_complement_checks=int((~audit.approved).sum()) if len(audit) else 0,
                                     newly_resolved_fine_blocks=fine_count,
                                     failed_fine_complement_checks=int((~fine_audit.approved).sum()),
                                     release_eligible=eligible, display_allowed=bool(eligible and load_data),
                                     reason=reason, signal_hypotheses=120))
            closure.append(dict(identity, **statistics, source='MC', approved=own_gate,
                                max_S_B=maximum, signal_hypotheses=120, display_allowed=True,
                                release_eligible=eligible, reason='MC_ALWAYS_VISIBLE'))
            blank = {field: finalize_sideband_validation_np.nan for field in finalize_sideband_validation_MEASUREMENT_FIELDS}
            blank.update(status='BLINDED', kappa_status='BLINDED', low_stat=False)
            if eligible and load_data:
                # Only approved source regions are dereferenced after every gate.
                data_values = {region: float(d_index[finalize_sideband_validation__key(*coordinate)][region])
                               for region, coordinate in coordinates.items()}
                data_variances = {region: float(d_index[finalize_sideband_validation__key(*coordinate)]['var_' + region])
                                  for region, coordinate in coordinates.items()}
                if not finalize_sideband_validation_np.isfinite(list(data_values.values()) + list(data_variances.values())).all():
                    raise AssertionError('An approved cached DATA source must contain finite measurements.')
                finalize_sideband_validation_np.testing.assert_array_equal(list(data_values.values()), list(data_variances.values()))
                blank = finalize_sideband_validation_statistics_from_regions(data_values, data_variances, config.get('min_neff', 10.))
            closure.append(dict(identity, **blank, source='DATA', approved=own_gate,
                                max_S_B=maximum, signal_hypotheses=120,
                                display_allowed=bool(eligible and load_data), release_eligible=eligible,
                                reason=reason if not eligible or load_data else 'DATA_EVALUATION_DISABLED'))
            if eligible:
                published.update(proposed)
                published_fine.update({name: finalize_sideband_validation_np.tile(mask, leaf_count) for name, mask in proposed.items()})
            for index, sample in enumerate(names):
                signal_rows.append(dict(identity, sample=sample,
                                        **{'S_' + r: signal_values[r][index] for r in finalize_sideband_validation_REGIONS},
                                        **{'var_S_' + r: signal_variances[r][index] for r in finalize_sideband_validation_REGIONS}))
            if spec['a_iso'] == spec['b_iso'] and spec['a_phi'] == spec['c_phi']:
                expected = b_index[finalize_sideband_validation__key(spec['a_iso'], spec['a_phi'])]
                fields = list(finalize_sideband_validation_REGIONS) + ['var_' + r for r in finalize_sideband_validation_REGIONS] + ['pred', 'pred_error', 'kappa', 'kappa_error']
                finalize_sideband_validation_np.testing.assert_allclose([statistics[f] for f in fields], [expected[f] for f in fields],
                                           rtol=1e-8, atol=1e-8, equal_nan=True)
                validation.append(dict(scope=scope + ':' + spec['candidate_id'],
                                       check='Zero-gap/ordinary boundary reproduces the source MC cache', passed=True))
                if eligible and load_data:
                    expected_data = d_index[finalize_sideband_validation__key(spec['a_iso'], spec['a_phi'])]
                    finalize_sideband_validation_np.testing.assert_allclose([blank[f] for f in fields], [expected_data[f] for f in fields],
                                               rtol=1e-8, atol=1e-8, equal_nan=True)
                    validation.append(dict(scope=scope + ':' + spec['candidate_id'],
                                           check='Approved ordinary DATA reproduces the source cache', passed=True))
            validation.append(dict(scope=scope + ':' + spec['candidate_id'],
                check='Candidate geometry reproduces cached ABCD sumw/sumw2 for background and all 120 signals', passed=True))
        validation.append(dict(scope=scope, check='Cumulative atomic MC and all 120 signal partitions conserve each source plane', passed=True))
    tables = dict(sbvr_closure=finalize_sideband_validation_pd.DataFrame(closure), sbvr_purity=finalize_sideband_validation_pd.DataFrame(purity),
                  sbvr_signal_points=finalize_sideband_validation_pd.DataFrame(signal_rows), sbvr_release_audit=finalize_sideband_validation_pd.DataFrame(release_rows),
                  sbvr_complements=finalize_sideband_validation_pd.DataFrame(complement_rows),
                  sbvr_input_manifest=finalize_sideband_validation_pd.DataFrame(inputs), sbvr_validation=finalize_sideband_validation_pd.DataFrame(validation))
    hidden = tables['sbvr_closure'].query("source == 'DATA' and not display_allowed")
    if not hidden[finalize_sideband_validation_MEASUREMENT_FIELDS].isna().all().all():
        raise AssertionError('Blocked VR DATA must not expose numeric statistics.')
    validation.extend([
        dict(scope='DATA', check='Every blocked DATA yield, prediction, ratio and uncertainty is NaN', passed=True),
        dict(scope='DATA', check='No original DATA, nominal DATA or high-mass DATA is read by VR validation', passed=True),
    ])
    tables['sbvr_validation'] = finalize_sideband_validation_pd.DataFrame(validation)
    return dict(tables=tables, metadata=dict(
        original_data_files_opened=0, nominal_data_evaluated=False, high_mass_data_evaluated=False,
        cached_data_evaluated=bool(load_data), signal_hypotheses=120,
        scopes=[x[0] for x in finalize_sideband_validation_SCOPE_DEFINITIONS],
        selection_policy='Candidates were chosen from nominal MC before VR DATA evaluation.',
        data_policy='Strict integrated all-120-model gate, approved source cache, and prospective disclosure audit.',
        disclosure_policy='Check newly identifiable cut-partition groups and nested/subset complements against previously displayed 09 results; preserve existing fine mass/prompt disclosures.',
        fine_mc_policy='Native 150–200/200–300 GeV partitions and PD prompt 3/ge4 leaves; sumw/sumw2 and all 120 signals reproduce established three-WP caches.',
        uncertainty='Disjoint weighted sumw2 propagation; sparse symmetric errors are diagnostic, not exact ratio intervals.',
        release_order=finalize_sideband_validation__release_order(selection).candidate_id.tolist()))


def finalize_sideband_validation__key(x, y):
    return round(float(x), 12), round(float(y), 12)


def finalize_sideband_validation__background_index(frame):
    if frame.duplicated(['tx', 'ty']).any():
        raise AssertionError('Each source plane requires unique cached coordinates.')
    return {finalize_sideband_validation__key(row['tx'], row['ty']): row for row in frame.to_dict('records')}


def finalize_sideband_validation__signal_index(frame):
    names = sorted(frame['sample'].unique())
    if len(names) != 120:
        raise AssertionError('Every VR must retain all 120 separate signal hypotheses.')
    result = {}
    for (tx, ty), group in frame.groupby(['tx', 'ty'], sort=False):
        group = group.set_index('sample').reindex(names)
        if len(group) != 120 or group.isna().any().any():
            # Only signal moments are required; cached ratios may be undefined.
            fields = ['S_' + r for r in finalize_sideband_validation_REGIONS] + ['var_S_' + r for r in finalize_sideband_validation_REGIONS]
            if len(group) != 120 or group[fields].isna().any().any():
                raise AssertionError('Incomplete signal moments in a source cache.')
        result[finalize_sideband_validation__key(tx, ty)] = group
    return result, names


def finalize_sideband_validation__load_caches(config, inputs, validation):
    result = {}
    for mass, key, bounds in [('low', 'vr1_output_dir', (0., 150.)),
                              ('intermediate', 'vr2_output_dir', (150., 300.))]:
        folder = finalize_sideband_validation_Path(config[key])
        manifest = folder / 'manifest.json'
        document = finalize_sideband_validation_json.loads(manifest.read_text())
        settings = document['settings']
        if (settings['topology'], settings['mass_min'], settings['mass_max'], settings['partner_iso_low'],
                settings['partner_iso_high'], settings['purity_limit']) != ('2mu2e', *bounds, 0., .5, .1):
            raise AssertionError('VR cache definitions differ from the common plane and fixed purity convention.')
        inputs.append(finalize_sideband_validation_common.candidate_input_record(manifest, 'Buffered VR source definitions'))
        frames = {}
        for filename, name in [('grid.csv', 'grid'), ('signal_points.csv', 'signals')]:
            path = folder / filename
            frames[name] = finalize_sideband_validation_pd.read_csv(path, float_precision='round_trip')
            inputs.append(finalize_sideband_validation_common.candidate_input_record(path, 'Buffered VR cached ' + name))
        result[mass] = frames
        validation.append(dict(scope=mass, check='Source cache mass, partner isolation, topology and 10% purity definitions match', passed=True))
    return result


def finalize_sideband_validation__ratios(signal, background):
    signal = finalize_sideband_validation_np.asarray(signal, float)
    return signal / background if background > 1e-8 else finalize_sideband_validation_np.full(signal.shape, finalize_sideband_validation_np.nan)


def finalize_sideband_validation__release_order(selection):
    # Prioritize the main A-preserving design; order is independent of VR DATA.
    def priority(row):
        if row['candidate_id'] == 'WP2':
            return 0
        if row['strategy'] == 'gap' and finalize_sideband_validation_np.isclose(row['b_iso'], .3) and finalize_sideband_validation_np.isclose(row['c_phi'], 2.):
            return 1
        if row['strategy'] == 'gap' and finalize_sideband_validation_np.isclose(row['b_iso'], .3) and finalize_sideband_validation_np.isclose(row['c_phi'], 1.5):
            return 2
        if row['strategy'] == 'loose' and finalize_sideband_validation_np.isclose(row['a_iso'], .3) and finalize_sideband_validation_np.isclose(row['a_phi'], 2.4):
            return 3
        return 4
    return selection.assign(_priority=[priority(row) for row in selection.to_dict('records')]).sort_values(
        ['_priority', 'candidate_id'], kind='stable').drop(columns='_priority')


def finalize_sideband_validation_cached_atomic_measurements(background, signals, x_edges, y_edges):
    """Recover disjoint MC rectangles from cached low-iso/low-angle cumulants.

    X spans [0,+inf); Y spans [-inf,+inf), retaining angle flow exactly as the
    source cache. No DATA field is used. Values and variances are differenced
    independently and checked against the source total for every signal model.
    """
    b = finalize_sideband_validation__background_index(background)
    s, names = finalize_sideband_validation__signal_index(signals)
    x_edges, y_edges = list(map(float, x_edges)), list(map(float, y_edges))
    if x_edges[0] != 0 or x_edges[-1] != finalize_sideband_validation_np.inf or y_edges[0] != -finalize_sideband_validation_np.inf or y_edges[-1] != finalize_sideband_validation_np.inf:
        raise ValueError('Atomic edges must cover the complete cached physical-isolation plane.')
    base_key = finalize_sideband_validation__key(.2, 2.7)
    def cumulative(x, y, signal=False, variance=False):
        if x == 0 or y == -finalize_sideband_validation_np.inf:
            return finalize_sideband_validation_np.zeros(120) if signal else 0.
        xx, yy = (.2 if x == finalize_sideband_validation_np.inf else x), (2.7 if y == finalize_sideband_validation_np.inf else y)
        row = (s if signal else b)[finalize_sideband_validation__key(xx, yy)]
        prefix = ('var_S_' if variance else 'S_') if signal else ('var_' if variance else '')
        regions = finalize_sideband_validation_REGIONS if x == finalize_sideband_validation_np.inf and y == finalize_sideband_validation_np.inf else ('C', 'D') if x == finalize_sideband_validation_np.inf else ('A', 'C') if y == finalize_sideband_validation_np.inf else ('C',)
        return finalize_sideband_validation_np.asarray(sum(row[prefix + region] for region in regions), float)
    records, values, variances, sig, sigvar = [], [], [], [], []
    for ix, (xl, xh) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        for iy, (yl, yh) in enumerate(zip(y_edges[:-1], y_edges[1:])):
            records.append(dict(atom_id=len(records), ix=ix, iy=iy, xlow=xl, xhigh=xh, ylow=yl, yhigh=yh))
            for output, is_signal, is_variance in [(values, False, False), (variances, False, True),
                                                  (sig, True, False), (sigvar, True, True)]:
                pieces = [cumulative(xh, yh, is_signal, is_variance), cumulative(xl, yh, is_signal, is_variance),
                          cumulative(xh, yl, is_signal, is_variance), cumulative(xl, yl, is_signal, is_variance)]
                value = pieces[0] - pieces[1] - pieces[2] + pieces[3]
                scale = finalize_sideband_validation_np.maximum(1., sum(finalize_sideband_validation_np.abs(piece) for piece in pieces))
                if finalize_sideband_validation_np.any(value < -1e-9 * scale) or not finalize_sideband_validation_np.isfinite(value).all():
                    raise AssertionError('Invalid cached MC cumulative difference.')
                output.append(finalize_sideband_validation_np.maximum(value, 0.))
    result = dict(atoms=finalize_sideband_validation_pd.DataFrame(records), values=finalize_sideband_validation_np.asarray(values, float), variances=finalize_sideband_validation_np.asarray(variances, float),
                  signals=finalize_sideband_validation_np.asarray(sig, float).T, signal_variances=finalize_sideband_validation_np.asarray(sigvar, float).T, signal_names=names)
    finalize_sideband_validation_np.testing.assert_allclose(result['values'].sum(), sum(b[base_key][r] for r in finalize_sideband_validation_REGIONS), rtol=1e-9, atol=1e-8)
    finalize_sideband_validation_np.testing.assert_allclose(result['variances'].sum(), sum(b[base_key]['var_' + r] for r in finalize_sideband_validation_REGIONS), rtol=1e-9, atol=1e-8)
    finalize_sideband_validation_np.testing.assert_allclose(result['signals'].sum(axis=1), sum(s[base_key]['S_' + r] for r in finalize_sideband_validation_REGIONS), rtol=1e-8, atol=1e-8)
    finalize_sideband_validation_np.testing.assert_allclose(result['signal_variances'].sum(axis=1), sum(s[base_key]['var_S_' + r] for r in finalize_sideband_validation_REGIONS), rtol=1e-8, atol=1e-8)
    return result


def finalize_sideband_validation_rectangular_mask(atoms, xlow, xhigh, ylow, yhigh):
    return ((atoms.xlow >= xlow) & (atoms.xhigh <= xhigh)
            & (atoms.ylow >= ylow) & (atoms.yhigh <= yhigh)).to_numpy(bool)


def finalize_sideband_validation_candidate_masks(spec, atoms):
    x, y, xb, yc = (float(spec[name]) for name in ('a_iso', 'a_phi', 'b_iso', 'c_phi'))
    if not (0 < x <= xb and yc <= y):
        raise ValueError('The ABCD rectangles must be disjoint and buffer-compatible.')
    return dict(A=finalize_sideband_validation_rectangular_mask(atoms, 0., x, y, finalize_sideband_validation_np.inf),
                B=finalize_sideband_validation_rectangular_mask(atoms, xb, finalize_sideband_validation_np.inf, y, finalize_sideband_validation_np.inf),
                C=finalize_sideband_validation_rectangular_mask(atoms, 0., x, -finalize_sideband_validation_np.inf, yc),
                D=finalize_sideband_validation_rectangular_mask(atoms, xb, finalize_sideband_validation_np.inf, -finalize_sideband_validation_np.inf, yc))


def finalize_sideband_validation__basis(rows, width):
    matrix = finalize_sideband_validation_np.asarray(list(rows), float)
    if matrix.size == 0:
        return finalize_sideband_validation_np.empty((0, width))
    matrix = finalize_sideband_validation_np.unique(matrix.reshape(-1, width), axis=0)
    with finalize_sideband_validation_threadpool_limits(limits=1, user_api='blas'):
        _, singular, vh = finalize_sideband_validation_np.linalg.svd(matrix, full_matrices=False)
    rank = int(finalize_sideband_validation_np.count_nonzero(singular > 1e-9))
    return vh[:rank]


def finalize_sideband_validation__identifiable(mask, basis):
    value = finalize_sideband_validation_np.asarray(mask, float)
    coefficients = finalize_sideband_validation_np.einsum('ij,j->i', basis, value)
    reconstructed = finalize_sideband_validation_np.einsum('ij,i->j', basis, coefficients)
    return bool(finalize_sideband_validation_np.max(finalize_sideband_validation_np.abs(value - reconstructed), initial=0.) < 1e-8)


def finalize_sideband_validation__prospective_patterns(old_masks, new_masks, width):
    old_basis = finalize_sideband_validation__basis(old_masks.values(), width)
    combined = {**old_masks, **new_masks}
    basis = finalize_sideband_validation__basis(combined.values(), width)
    rows = finalize_sideband_validation_np.asarray(list(combined.values()), bool)
    labels = list(combined)
    patterns = {}
    def add(mask, kind, description):
        mask = finalize_sideband_validation_np.asarray(mask, bool)
        if not mask.any() or finalize_sideband_validation__identifiable(mask, old_basis) or not finalize_sideband_validation__identifiable(mask, basis):
            return
        token = finalize_sideband_validation_np.packbits(mask).tobytes()
        patterns.setdefault(token, dict(mask=mask, audit_kind=kind, description=description))
    if len(rows):
        _, inverse = finalize_sideband_validation_np.unique(rows.T, axis=0, return_inverse=True)
        for group in finalize_sideband_validation_np.unique(inverse):
            mask = inverse == group
            add(mask, 'IDENTIFIABLE_PARTITION_BLOCK', f'membership group {int(group)}')
        for name, mask in new_masks.items():
            add(mask, 'NEW_REGION', name)
        for i, outer in enumerate(rows):
            for j, inner in enumerate(rows):
                if i != j and finalize_sideband_validation_np.all(~inner | outer):
                    add(outer & ~inner, 'NESTED_REGION_COMPLEMENT', labels[i] + ' minus ' + labels[j])
            # Remove all known pairwise-disjoint contained regions. This catches
            # the unreported buffer left after multiple new ABCD pieces.
            contained = [(labels[j], inner) for j, inner in enumerate(rows)
                         if i != j and inner.any() and finalize_sideband_validation_np.all(~inner | outer) and not finalize_sideband_validation_np.array_equal(inner, outer)]
            used = finalize_sideband_validation_np.zeros(width, bool)
            removed = []
            for name, inner in sorted(contained, key=lambda item: (-int(item[1].sum()), item[0])):
                if not (used & inner).any():
                    used |= inner
                    removed.append(name)
            if removed:
                add(outer & ~used, 'DISJOINT_SUBSET_REMAINDER', labels[i] + ' minus ' + ','.join(removed))
    return list(patterns.values())


def finalize_sideband_validation_disclosure_audit(old_masks, new_masks, atomic):
    """Audit new identifiable partition blocks and exact published complements.

    Previously identifiable quantities are explicitly grandfathered: an unsafe
    historical difference is not treated as new information from this report.
    No observed DATA yield is required to decide whether a complement is safe.
    """
    if not isinstance(old_masks, dict):
        old_masks = {f'old:{i}': finalize_sideband_validation_np.asarray(mask, bool) for i, mask in enumerate(old_masks)}
    if not isinstance(new_masks, dict):
        new_masks = {f'new:{i}': finalize_sideband_validation_np.asarray(mask, bool) for i, mask in enumerate(new_masks)}
    records = []
    for pattern in finalize_sideband_validation__prospective_patterns(old_masks, new_masks, len(atomic['values'])):
        mask = pattern.pop('mask')
        background = float(atomic['values'][mask].sum())
        variance = float(atomic['variances'][mask].sum())
        signal = atomic['signals'][:, mask].sum(axis=1)
        ratios = finalize_sideband_validation__ratios(signal, background)
        valid = bool(background > 1e-8 and finalize_sideband_validation_np.isfinite(ratios).all())
        worst = int(finalize_sideband_validation_np.argmax(signal))
        maximum = float(ratios[worst]) if valid else finalize_sideband_validation_np.nan
        records.append(dict(**pattern, atomic_cells=int(mask.sum()), background=background,
                            var_background=variance, max_S_B=maximum,
                            worst_sample=atomic['signal_names'][worst], signal_hypotheses=120,
                            approved=bool(valid and maximum < .1),
                            reason='PASS' if valid and maximum < .1 else 'NONPOSITIVE_BACKGROUND' if not valid else 'CONTAMINATION_AT_OR_ABOVE_10_PERCENT'))
    return finalize_sideband_validation_pd.DataFrame(records, columns=['audit_kind', 'description', 'atomic_cells', 'background', 'var_background',
                                         'max_S_B', 'worst_sample', 'signal_hypotheses', 'approved', 'reason'])


def finalize_sideband_validation__existing_observations(tables, mass, state, atoms):
    if mass == 'intermediate' and state == 'PP':
        leaves = [(150., 200., 'all'), (200., 300., 'all')]
    elif mass == 'intermediate' and state == 'PD':
        leaves = [(low, high, prompt) for low, high in [(150., 200.), (200., 300.)] for prompt in ('3', 'ge4')]
    else:
        leaves = [(0., 150., 'all')] if mass == 'low' else [(150., 300., 'all')]
    count = len(leaves)
    prior, fine = {}, {}
    source = tables['report_lowmass_dd'] if state == 'DD' else tables['mass_closure']
    selected = source.loc[source.source.eq('DATA') & source.display_allowed & source.plane.eq(state)]
    if state != 'DD':
        selected = selected.loc[selected.mass_slice.eq(mass)]
    for row in selected.itertuples():
        spec = dict(a_iso=row.tx, b_iso=row.tx, a_phi=row.ty, c_phi=row.ty)
        for region, mask in finalize_sideband_validation_candidate_masks(spec, atoms).items():
            name = 'previous:' + row.wp + ':' + region
            prior[name] = mask
            fine[name] = finalize_sideband_validation_np.tile(mask, count)
    if count > 1:
        selected = tables['strata_closure'].loc[
            tables['strata_closure'].source.eq('DATA') & tables['strata_closure'].display_allowed
            & tables['strata_closure'].plane.eq(state) & tables['strata_closure'].mass_slice.eq(mass)]
        for row in selected.itertuples():
            spec = dict(a_iso=row.tx, b_iso=row.tx, a_phi=row.ty, c_phi=row.ty)
            leaf_indices = [i for i, (low, high, prompt) in enumerate(leaves)
                            if low >= row.mass_low and high <= row.mass_high
                            and (row.prompt_group == 'all' or row.prompt_group == prompt)]
            for region, mask in finalize_sideband_validation_candidate_masks(spec, atoms).items():
                combined = finalize_sideband_validation_np.zeros((count, len(atoms)), bool)
                combined[leaf_indices] = mask
                fine['previous:' + row.selection_id + ':' + row.wp + ':' + region] = combined.ravel()
    return prior, fine, count


def finalize_sideband_validation__validated_strata_mc(config, tables, supplied, inputs, validation):
    _settings = _FINALIZE_NAMESPACES['abcd_final_check_strata']._settings; _specs = _FINALIZE_NAMESPACES['abcd_final_check_strata']._specs; _load_mc = _FINALIZE_NAMESPACES['abcd_final_check_strata']._load_mc
    mc = _load_mc(_settings(config), _specs()) if supplied is None else supplied
    inputs.extend(mc.get('manifest', []))
    validation.extend(mc.get('validation', []))
    selected = tables['strata_closure'].loc[tables['strata_closure'].source.eq('MC')]
    for row in selected.itertuples():
        actual = finalize_sideband_validation_common._comparison_regions(mc['backgrounds'][row.selection_id], row.tx, row.ty)
        for region in finalize_sideband_validation_REGIONS:
            finalize_sideband_validation_np.testing.assert_allclose([actual[region]['value'], actual[region]['variance']],
                [getattr(row, region), getattr(row, 'var_' + region)], rtol=1e-8, atol=1e-8)
    for row in tables['strata_signal_points'].itertuples():
        actual = finalize_sideband_validation_common._comparison_regions(mc['signals'][row.selection_id][row.sample], row.tx, row.ty)
        for region in finalize_sideband_validation_REGIONS:
            finalize_sideband_validation_np.testing.assert_allclose([actual[region]['value'], actual[region]['variance']],
                [getattr(row, 'S_' + region), getattr(row, 'var_S_' + region)], rtol=1e-8, atol=1e-8)
    validation.append(dict(scope='Fine MC',
        check='All native mass/prompt MC planes reproduce prior Ref/WP1/WP2 backgrounds and 120 signal hypotheses', passed=True))
    return mc


def finalize_sideband_validation__native_atomic_values(plane, atoms, field):
    def bound(edges, value):
        return None if not finalize_sideband_validation_np.isfinite(value) else finalize_sideband_validation_common._comparison_split(edges, value)
    return finalize_sideband_validation_np.asarray([plane[field][
        slice(bound(plane['xedges'], row.xlow), bound(plane['xedges'], row.xhigh)),
        slice(bound(plane['yedges'], row.ylow), bound(plane['yedges'], row.yhigh))].sum()
        for row in atoms.itertuples()], float)


def finalize_sideband_validation__fine_atomic_measurements(mc, mass, state, coarse, validation):
    if mass != 'intermediate' or state not in ('PP', 'PD'):
        raise ValueError('Fine public observations are established only in intermediate PP/PD.')
    leaves = [f'{state}_{low}_{high}_{prompt}' for low, high in [(150, 200), (200, 300)]
              for prompt in (('all',) if state == 'PP' else ('3', 'ge4'))]
    names = coarse['signal_names']
    atoms = coarse['atoms']
    backgrounds = [mc['backgrounds'][name] for name in leaves]
    result = dict(atoms=finalize_sideband_validation_pd.concat([atoms.assign(leaf=name) for name in leaves], ignore_index=True),
        values=finalize_sideband_validation_np.concatenate([finalize_sideband_validation__native_atomic_values(plane, atoms, 'values') for plane in backgrounds]),
        variances=finalize_sideband_validation_np.concatenate([finalize_sideband_validation__native_atomic_values(plane, atoms, 'variances') for plane in backgrounds]),
        signals=finalize_sideband_validation_np.asarray([finalize_sideband_validation_np.concatenate([finalize_sideband_validation__native_atomic_values(mc['signals'][leaf][sample], atoms, 'values')
                            for leaf in leaves]) for sample in names]),
        signal_variances=finalize_sideband_validation_np.asarray([finalize_sideband_validation_np.concatenate([finalize_sideband_validation__native_atomic_values(mc['signals'][leaf][sample], atoms, 'variances')
                            for leaf in leaves]) for sample in names]), signal_names=names)
    for field in ('values', 'variances', 'signals', 'signal_variances'):
        shape = (len(leaves), len(atoms)) if field in ('values', 'variances') else (120, len(leaves), len(atoms))
        axis = 0 if len(shape) == 2 else 1
        finalize_sideband_validation_np.testing.assert_allclose(result[field].reshape(shape).sum(axis=axis), coarse[field], rtol=1e-8, atol=1e-8)
        if finalize_sideband_validation_np.any(result[field] < -1e-8) or not finalize_sideband_validation_np.isfinite(result[field]).all():
            raise AssertionError('Invalid native fine-partition MC moments.')
    validation.append(dict(scope=mass + '_' + state,
        check='Native fine mass/prompt atoms exactly reproduce coarse cache moments for background and every signal', passed=True))
    return result


# ========================================================================
# abcd_final_check_strata: original study API
# ========================================================================
"""Fixed-WP mass and prompt-displacement checks for notebook 09.

Only explicit PP, PD and DP selections below 300 GeV are defined.  The MC
purity gate treats the 120 signal hypotheses separately.  DATA access is a
separate, opt-in step after the complete complementary-family gate is known.
"""
from pathlib import Path as finalize_final_check_strata_Path
import gc as finalize_final_check_strata_gc

import numpy as finalize_final_check_strata_np
import pandas as finalize_final_check_strata_pd

finalize_final_check_strata_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_final_check_strata_POINTS = _FINALIZE_NAMESPACES['abcd_final_check'].POINTS; finalize_final_check_strata_NUMERIC_STATS = _FINALIZE_NAMESPACES['abcd_final_check'].NUMERIC_STATS
finalize_final_check_strata__state_index = _FINALIZE_NAMESPACES['abcd_mass_inclusive']._state_index


finalize_final_check_strata_FINE_MASSES = ((0., 50.), (50., 100.), (100., 150.), (150., 200.), (200., 300.))
finalize_final_check_strata_COARSE_MASSES = (('low', 0., 150.), ('intermediate', 150., 300.))
finalize_final_check_strata_STATES = ('PP', 'PD', 'DP')
finalize_final_check_strata_RATIO_FIELDS = ('R_high', 'R_high_error', 'R_low', 'R_low_error')
finalize_final_check_strata_DATA_NUMERIC_STATS = finalize_final_check_strata_NUMERIC_STATS + list(finalize_final_check_strata_RATIO_FIELDS)


def finalize_final_check_strata__settings(config):
    if config['topology'] != '2mu2e' or config['purity_limit'] != .1:
        raise ValueError('Stratified validation requires 2mu2e and strict S/B < 0.10.')
    if (config.get('partner_iso_low', 0.), config.get('partner_iso_high', .5)) != (0., .5):
        raise ValueError('Partner isolation must remain in [0, 0.5).')
    actual = [(p['wp'], p['tx'], p['ty']) for p in config['working_points']]
    if actual != list(finalize_final_check_strata_POINTS):
        raise ValueError('Only the three established fixed working points are allowed.')
    background = finalize_final_check_strata_Path(config['bkg_dir'])
    if '_bkg_' not in background.name:
        raise ValueError('Expected the established background production directory.')
    return dict(config, bkg_dir=background,
                signal_dir=background.with_name(background.name.replace('_bkg_', '_signal_')),
                data_dir=background.with_name(background.name.replace('_bkg_', '_data_')))


def finalize_final_check_strata__specs():
    result = {}
    for low, high in finalize_final_check_strata_FINE_MASSES:
        for state, prompt in [('PP', 'all'), ('PD', 'all'), ('DP', 'all'), ('PD', '3'), ('PD', 'ge4')]:
            name = f'{state}_{int(low)}_{int(high)}_{prompt}'
            mass_slice = 'low' if high <= 150 else 'intermediate'
            result[name] = dict(selection_id=name, level='fine', plane=state,
                                mass_low=low, mass_high=high, mass_label=f'{low:g}–{high:g}',
                                mass_slice=mass_slice, prompt_group=prompt,
                                family=f'{mass_slice}_{state}')
    for mass_slice, low, high in finalize_final_check_strata_COARSE_MASSES:
        for prompt in ('3', 'ge4'):
            name = f'PD_{int(low)}_{int(high)}_{prompt}'
            result[name] = dict(selection_id=name, level='coarse', plane='PD',
                                mass_low=low, mass_high=high, mass_label=f'{low:g}–{high:g}',
                                mass_slice=mass_slice, prompt_group=prompt,
                                family=f'{mass_slice}_PD')
    return result


def finalize_final_check_strata__sum_planes(planes):
    planes = list(planes)
    if not planes:
        raise ValueError('A nonempty list of planes is required.')
    for plane in planes[1:]:
        for field in ('xedges', 'yedges'):
            finalize_final_check_strata_np.testing.assert_array_equal(plane[field], planes[0][field])
    return dict(finalize_final_check_strata_common._comparison_plane(
        sum(plane['values'] for plane in planes), sum(plane['variances'] for plane in planes),
        planes[0]['xedges'], planes[0]['yedges']), kind='mu_dphi')


def finalize_final_check_strata__selection_index(edges, spec, config):
    if spec['plane'] not in finalize_final_check_strata_STATES or not (0 <= spec['mass_low'] < spec['mass_high'] <= 300):
        raise ValueError('Only explicit PP/PD/DP VRs within 0–300 GeV are permitted.')
    index = list(finalize_final_check_strata__state_index(edges, spec['plane'], config))
    if spec['prompt_group'] != 'all':
        if spec['plane'] != 'PD' or spec['prompt_group'] not in ('3', 'ge4'):
            raise ValueError('Prompt-pixel subdivision is defined only for PD.')
        split = finalize_final_check_strata_common._comparison_split(edges['disp0'], 3.5)
        index[2] = (slice(finalize_final_check_strata_common._comparison_split(edges['disp0'], 2.5), split)
                    if spec['prompt_group'] == '3' else slice(split, None))
    index.append(slice(finalize_final_check_strata_common._comparison_split(edges['ljlj_mass'], spec['mass_low']),
                       finalize_final_check_strata_common._comparison_split(edges['ljlj_mass'], spec['mass_high'])))
    return tuple(index)


def finalize_final_check_strata__project_mc(weights, variances, edges, specs, config):
    """Reduce each displacement group once, preserving native mass partitions."""
    result = {}
    for state, prompt in [('PP', 'all'), ('PD', 'all'), ('DP', 'all'), ('PD', '3'), ('PD', 'ge4')]:
        group = [s for s in specs.values() if s['plane'] == state and s['prompt_group'] == prompt
                 and s['level'] == 'fine']
        index = list(finalize_final_check_strata__selection_index(edges, group[0], config))
        # MC only: keep the full physical 0–300 interval until its native bins are split.
        mass_start = finalize_final_check_strata_common._comparison_split(edges['ljlj_mass'], 0.)
        index[-1] = slice(mass_start, finalize_final_check_strata_common._comparison_split(edges['ljlj_mass'], 300.))
        selected = [array[tuple(index)] for array in (weights, variances)]
        finalize_final_check_strata_common._comparison_validate_arrays(*selected)
        reduced = [array.sum(axis=(1, 2, 3)) for array in selected]
        for spec in group:
            low = finalize_final_check_strata_common._comparison_split(edges['ljlj_mass'], spec['mass_low']) - mass_start
            high = finalize_final_check_strata_common._comparison_split(edges['ljlj_mass'], spec['mass_high']) - mass_start
            arrays = [array[:, :, low:high].sum(axis=2) for array in reduced]
            result[spec['selection_id']] = dict(finalize_final_check_strata_common._comparison_plane(
                *arrays, edges['iso0'], edges['abs_dphi']), kind='mu_dphi')
        summed = finalize_final_check_strata__sum_planes(result[s['selection_id']] for s in group)
        for field, original in zip(('values', 'variances'), reduced):
            finalize_final_check_strata_np.testing.assert_allclose(summed[field], original.sum(axis=2), rtol=1e-10, atol=1e-8)
    for spec in specs.values():
        if spec['level'] == 'coarse':
            result[spec['selection_id']] = finalize_final_check_strata__sum_planes(
                result[s['selection_id']] for s in specs.values()
                if s['level'] == 'fine' and s['plane'] == 'PD'
                and s['prompt_group'] == spec['prompt_group'] and s['mass_slice'] == spec['mass_slice'])
    for low, high in finalize_final_check_strata_FINE_MASSES:
        parent = result[f'PD_{int(low)}_{int(high)}_all']
        children = finalize_final_check_strata__sum_planes(result[f'PD_{int(low)}_{int(high)}_{prompt}'] for prompt in ('3', 'ge4'))
        for field in ('values', 'variances'):
            finalize_final_check_strata_np.testing.assert_allclose(parent[field], children[field], rtol=1e-10, atol=1e-8)
    return result


def finalize_final_check_strata__load_mc(config, specs):
    backgrounds = sorted(config['bkg_dir'].glob('*.coffea'))
    if len(backgrounds) != 18:
        raise ValueError('Expected exactly 18 background MC samples.')
    signals = finalize_final_check_strata_common.candidate_signal_paths(config)
    totals = {key: None for key in specs}
    samples = {key: {} for key in specs}
    signal_planes = {key: {} for key in specs}
    manifest, validation = [], []
    jobs = [(path, 'Background') for path in backgrounds] + [(path, 'Signal hypothesis') for path in signals]
    for index, (path, source) in enumerate(jobs, 1):
        loaded, payload = finalize_final_check_strata_common._comparison_payload(path, 'sixd')
        weights, variances, edges = finalize_final_check_strata_common._comparison_current_cube(payload, path, config)
        projected = finalize_final_check_strata__project_mc(weights, variances, edges, specs, config)
        for name, plane in projected.items():
            if source == 'Background':
                samples[name][path.stem] = plane
                totals[name] = finalize_final_check_strata_common.add_candidate(totals[name], plane, finalize_final_check_strata_common._comparison_group(path.stem))
            else:
                signal_planes[name][path.stem] = plane
        manifest.append(finalize_final_check_strata_common.candidate_input_record(path, 'Strata ' + source))
        validation.append(dict(scope=path.stem, check='Native fine mass and PD prompt partitions conserve sumw/sumw2',
                               passed=True))
        del loaded, payload, weights, variances, projected
        finalize_final_check_strata_gc.collect()
        if index % 20 == 0 or index == len(jobs):
            print(f'09 strata MC {index}/{len(jobs)}', flush=True)
    return dict(backgrounds=totals, samples=samples, signals=signal_planes,
                manifest=manifest, validation=validation)


def finalize_final_check_strata__parent_plane(mc, specs, state, mass_slice, sample=None):
    keys = [s['selection_id'] for s in specs.values() if s['level'] == 'fine'
            and s['plane'] == state and s['prompt_group'] == 'all' and s['mass_slice'] == mass_slice]
    return finalize_final_check_strata__sum_planes(mc['backgrounds'][key] if sample is None else mc['signals'][key][sample] for key in keys)


def finalize_final_check_strata__validate_parents(mc, specs, config, base_result):
    validation, manifest = [], []
    for state in finalize_final_check_strata_STATES:
        for mass_slice, _, _ in finalize_final_check_strata_COARSE_MASSES:
            actual = finalize_final_check_strata__parent_plane(mc, specs, state, mass_slice)
            expected = base_result['planes']['mass'][state][mass_slice]
            for field in ('values', 'variances'):
                finalize_final_check_strata_np.testing.assert_allclose(actual[field], expected[field], rtol=1e-9, atol=1e-8)
            validation.append(dict(scope=f'{mass_slice}_{state}',
                                   check='Fine mass background sum reproduces the original 09 parent plane', passed=True))
    for mass_slice, directory in [('low', config['vr1_output_dir']), ('intermediate', config['vr2_output_dir'])]:
        path = finalize_final_check_strata_Path(directory) / 'signal_points.csv'
        cached = finalize_final_check_strata_pd.read_csv(path, float_precision='round_trip')
        cached = cached.loc[cached.plane.isin(finalize_final_check_strata_STATES)]
        points = finalize_final_check_strata_pd.DataFrame(finalize_final_check_strata_POINTS, columns=['wp', 'tx', 'ty'])
        cached = points.merge(cached, on=['tx', 'ty'], validate='one_to_many')
        if len(cached) != 3 * 3 * 120:
            raise AssertionError('Every parent cache must cover 120 hypotheses at each fixed WP and state.')
        parent_signals = {state: {sample: finalize_final_check_strata__parent_plane(mc, specs, state, mass_slice, sample)
                                  for sample in sorted(next(iter(mc['signals'].values())))} for state in finalize_final_check_strata_STATES}
        for row in cached.itertuples():
            actual = finalize_final_check_strata_common._comparison_regions(parent_signals[row.plane][row.sample], row.tx, row.ty)
            for region in 'ABCD':
                finalize_final_check_strata_np.testing.assert_allclose([actual[region]['value'], actual[region]['variance']],
                                           [getattr(row, 'S_' + region), getattr(row, 'var_S_' + region)],
                                           rtol=1e-8, atol=1e-8)
        validation.append(dict(scope=mass_slice,
                               check='All 120 signal parent sumw/sumw2 reproduce 01/03 at all three fixed WPs', passed=True))
        manifest.append(finalize_final_check_strata_common.candidate_input_record(path, 'Strata signal parent regression cache'))
    return validation, manifest


def finalize_final_check_strata__ratio(value, denominator, variance, denominator_variance):
    if denominator <= 0 or not finalize_final_check_strata_np.all(finalize_final_check_strata_np.isfinite([value, denominator, variance, denominator_variance])):
        return finalize_final_check_strata_np.nan, finalize_final_check_strata_np.nan
    ratio = value / denominator
    error = finalize_final_check_strata_np.sqrt(variance / denominator**2 + value**2 * denominator_variance / denominator**4)
    return float(ratio), float(error)


def finalize_final_check_strata__statistics(plane, tx, ty, min_neff):
    row = finalize_final_check_strata_common.plane_statistics(plane, tx, ty, min_neff=min_neff)
    row['R_high'], row['R_high_error'] = finalize_final_check_strata__ratio(row['A'], row['B'], row['var_A'], row['var_B'])
    row['R_low'], row['R_low_error'] = finalize_final_check_strata__ratio(row['C'], row['D'], row['var_C'], row['var_D'])
    if finalize_final_check_strata_np.isfinite(row['kappa']) and finalize_final_check_strata_np.isfinite(row['R_high']) and row['R_low'] > 0:
        finalize_final_check_strata_np.testing.assert_allclose(row['R_high'] / row['R_low'], row['kappa'], rtol=1e-12, atol=1e-12)
    return row


def finalize_final_check_strata__purity_tables(mc, specs):
    regions, gates, signal_rows = [], [], []
    for name, spec in specs.items():
        signals = mc['signals'][name]
        if len(signals) != 120:
            raise AssertionError('Every stratum requires all 120 independent signal hypotheses.')
        for wp, tx, ty in finalize_final_check_strata_POINTS:
            identity = dict(spec, wp=wp, tx=tx, ty=ty)
            background = finalize_final_check_strata_common._comparison_regions(mc['backgrounds'][name], tx, ty)
            signal_values = {sample: finalize_final_check_strata_common._comparison_regions(plane, tx, ty) for sample, plane in signals.items()}
            regional = []
            for region in 'ABCD':
                b, vb = background[region]['value'], background[region]['variance']
                valid_b = bool(finalize_final_check_strata_np.isfinite(b) and b > 1e-8)
                values = [(sample, item[region]['value'], item[region]['variance'])
                          for sample, item in signal_values.items()]
                valid_s = all(finalize_final_check_strata_np.isfinite(s) and s >= 0 and finalize_final_check_strata_np.isfinite(vs) and vs >= 0 for _, s, vs in values)
                worst, s, vs = max(values, key=lambda item: item[1])
                ratio, error = finalize_final_check_strata__ratio(s, b, vs, vb) if valid_b and valid_s else (finalize_final_check_strata_np.nan, finalize_final_check_strata_np.nan)
                regional.append(dict(identity, region=region, background=b, var_background=vb,
                                     background_error=finalize_final_check_strata_np.sqrt(vb), signal_max=s, var_signal_max=vs,
                                     max_S_B=ratio, max_S_B_error=error, worst_sample=worst,
                                     valid_background=valid_b, valid_signals=bool(valid_s), signal_hypotheses=120,
                                     approved=bool(valid_b and valid_s and ratio < .1)))
            regions.extend(regional)
            complete = sum(finalize_final_check_strata_np.isfinite(item[r]['value']) and item[r]['value'] >= 0
                           for item in signal_values.values() for r in 'ABCD')
            approved = all(item['approved'] for item in regional) and complete == 480
            max_sb = max(item['max_S_B'] for item in regional) if all(item['valid_background'] and item['valid_signals'] for item in regional) else finalize_final_check_strata_np.nan
            reason = ('PURITY_PASS' if approved else 'NONPOSITIVE_BACKGROUND'
                      if not all(item['valid_background'] for item in regional) else 'INVALID_SIGNAL'
                      if not all(item['valid_signals'] for item in regional) else 'PURITY_AT_OR_ABOVE_10_PERCENT')
            gates.append(dict(identity, approved=approved, max_sb=max_sb,
                              signal_hypotheses=120, complete_signal_cells=complete, own_gate_reason=reason))
            for sample, values in signal_values.items():
                signal_rows.append(dict(identity, sample=sample,
                                        **{'S_' + r: values[r]['value'] for r in 'ABCD'},
                                        **{'var_S_' + r: values[r]['variance'] for r in 'ABCD'}))
    gates = finalize_final_check_strata_pd.DataFrame(gates)
    # All three WPs belong to one release decision: no partly released family can
    # be subtracted from an already displayed coarse parent to recover a vetoed child.
    for family in gates.family.unique():
        member = gates.family.eq(family)
        leaves = member & gates.level.eq('fine')
        leaves &= (gates.prompt_group.isin(['3', 'ge4']) if family.endswith('_PD') else gates.prompt_group.eq('all'))
        family_approved = bool(gates.loc[leaves, 'approved'].all())
        gates.loc[member, 'family_approved'] = family_approved
        gates.loc[member, 'family_leaf_points'] = int(leaves.sum())
        gates.loc[member, 'family_blocked_leaf_points'] = int((~gates.loc[leaves, 'approved']).sum())
    gates['family_approved'] = gates.family_approved.astype(bool)
    gates['display_allowed'] = gates.approved & gates.family_approved
    gates['reason'] = finalize_final_check_strata_np.where(gates.display_allowed, 'PURITY_AND_COMPLEMENTARY_FAMILY_PASS',
                              finalize_final_check_strata_np.where(~gates.approved, gates.own_gate_reason, 'COMPLEMENTARY_FAMILY_VETO'))
    return finalize_final_check_strata_pd.DataFrame(regions), gates, finalize_final_check_strata_pd.DataFrame(signal_rows)


def finalize_final_check_strata__closure_table(mc, specs, gates, config):
    rows = []
    for gate in gates.to_dict('records'):
        name = gate['selection_id']
        row = finalize_final_check_strata__statistics(mc['backgrounds'][name], gate['tx'], gate['ty'], config.get('min_neff', 10.))
        rows.append(dict(gate, **row, source='MC', display_allowed=True, reason='MC_ALWAYS_VISIBLE'))
        blank = finalize_final_check_strata_common._extension_blank_statistics(gate['tx'], gate['ty'], reason='DATA_NOT_LOADED')
        blank.update({field: finalize_final_check_strata_np.nan for field in finalize_final_check_strata_RATIO_FIELDS})
        rows.append(dict(gate, **blank, source='DATA', display_allowed=False,
                         reason='DATA_NOT_LOADED' if gate['display_allowed'] else gate['reason']))
    return finalize_final_check_strata_pd.DataFrame(rows)


def finalize_final_check_strata__composition_table(mc, specs):
    rows = []
    for name, spec in specs.items():
        for wp, tx, ty in finalize_final_check_strata_POINTS:
            totals = finalize_final_check_strata_common._comparison_regions(mc['backgrounds'][name], tx, ty)
            for sample, plane in mc['samples'][name].items():
                regions = finalize_final_check_strata_common._comparison_regions(plane, tx, ty)
                for region, item in regions.items():
                    total = totals[region]['value']
                    rows.append(dict(spec, wp=wp, tx=tx, ty=ty, source='MC', sample=sample,
                                     component=finalize_final_check_strata_common._comparison_group(sample), region=region,
                                     **{'yield': item['value']}, variance=item['variance'], error=item['error'],
                                     neff=item['neff'], fraction=item['value'] / total if total > 0 else finalize_final_check_strata_np.nan))
    return finalize_final_check_strata_pd.DataFrame(rows)


def finalize_final_check_strata__validate_output(result):
    table = result['tables']['strata_closure']
    blocked = table.source.eq('DATA') & ~table.display_allowed
    if not table.loc[blocked, finalize_final_check_strata_DATA_NUMERIC_STATS].isna().all().all():
        raise AssertionError('Blocked DATA must not contain numeric measurements.')
    if set(table.plane) != set(finalize_final_check_strata_STATES) or (table.mass_high > 300).any():
        raise AssertionError('A forbidden DATA selection entered the strata tables.')
    rows = result['tables']['strata_validation'].to_dict('records')
    rows.extend([
        dict(scope='strata', check='Blocked DATA measurements are NaN; only explicit PP/PD/DP below 300 GeV exist', passed=True),
        dict(scope='strata', check='R_high/R_low equals kappa wherever both are defined', passed=True),
    ])
    result['tables']['strata_validation'] = finalize_final_check_strata_pd.DataFrame(rows)


def finalize_final_check_strata_run_stratified_checks(config, base_result, load_data=True):
    """Calculate MC gates, then optionally project only approved DATA families."""
    config = finalize_final_check_strata__settings(config)
    specs = finalize_final_check_strata__specs()
    mc = finalize_final_check_strata__load_mc(config, specs)
    validation, manifest = finalize_final_check_strata__validate_parents(mc, specs, config, base_result)
    mc['validation'].extend(validation)
    mc['manifest'].extend(manifest)
    purity, gates, signals = finalize_final_check_strata__purity_tables(mc, specs)
    result = dict(config=config, specs=specs, planes=mc, tables=dict(
        strata_closure=finalize_final_check_strata__closure_table(mc, specs, gates, config),
        strata_purity_regions=purity, strata_gate_audit=gates,
        strata_signal_points=signals, strata_composition=finalize_final_check_strata__composition_table(mc, specs),
        strata_input_manifest=finalize_final_check_strata_pd.DataFrame(mc['manifest']), strata_validation=finalize_final_check_strata_pd.DataFrame(mc['validation'])),
        metadata=dict(signal_hypotheses=120, background_files=18, original_data_files_opened=0,
                      data_loaded=False, nominal_data_projected=False, nominal_data_evaluated=False,
                      high_mass_data_projected=False, fixed_working_points=list(finalize_final_check_strata_POINTS),
                      selections=len(specs), fine_selections=25, coarse_pd_child_selections=4,
                      purity_rule='All 120 signal hypotheses in every ABCD region: strictly S/B_MC < 0.10; background > 1e-8.',
                      complementary_rule='Release a coarse-family subdivision only if every finest leaf passes at all three fixed WPs.',
                      displacement='P mu: disp0 >= 2.5; PD children: [2.5,3.5) and [3.5,+inf), retaining overflow. EGM D: disp1 >= 0.5.',
                      flows='Partner iso [0,0.5); mu iso underflow excluded from ABCD, overflow retained; both angle flows retained; mass [0,300).',
                      ratio_error='sqrt(var(numerator)/denominator^2 + numerator^2*var(denominator)/denominator^4); denominator must be positive.',
                      ratio_interpretation='R_high=A/B and R_low=C/D; disjoint Poisson/weighted-MC regions, first-order sumw2 propagation.',
                      inference_limit='Diagnostic ratios and kappa are not a likelihood significance or a model of low-count ratio tails.'))
    finalize_final_check_strata__validate_output(result)
    if load_data:
        finalize_final_check_strata_finalize_stratified_data(config, result, base_result)
    return result


def finalize_final_check_strata__project_data_leaf(weights, variances, edges, spec, config):
    """Access one approved explicit finest leaf, never a coarse or DD selection."""
    if spec['level'] != 'fine' or (spec['plane'] == 'PD' and spec['prompt_group'] == 'all'):
        raise ValueError('DATA must be projected in the finest mass × prompt leaves only.')
    index = finalize_final_check_strata__selection_index(edges, spec, config)
    arrays = [array[index] for array in (weights, variances)]
    finalize_final_check_strata_common._comparison_validate_arrays(*arrays)
    reduced = [array.sum(axis=(1, 2, 3, 5)) for array in arrays]
    for projected, selected in zip(reduced, arrays):
        finalize_final_check_strata_np.testing.assert_allclose(projected.sum(), selected.sum(), rtol=1e-10, atol=1e-8)
    return dict(finalize_final_check_strata_common._comparison_plane(*reduced, edges['iso0'], edges['abs_dphi']), kind='mu_dphi')


def finalize_final_check_strata__load_approved_data(config, specs, gates):
    allowed = []
    for name, selected in gates.groupby('selection_id', sort=False):
        if len(selected) != 3:
            raise AssertionError('Every DATA selection requires the same three fixed WP gates.')
        if selected.display_allowed.all():
            allowed.append(name)
        elif selected.display_allowed.any():
            raise AssertionError('A DATA selection cannot be released at only a subset of WPs.')
    leaves = [name for name in allowed if specs[name]['level'] == 'fine'
              and (specs[name]['plane'] != 'PD' or specs[name]['prompt_group'] != 'all')]
    data, manifest = {name: None for name in leaves}, []
    if not leaves:
        return data, manifest, []
    paths = sorted(config['data_dir'].glob('*.coffea'))
    if len(paths) != 68:
        raise ValueError('Expected the established 68 DATA files.')
    for index, path in enumerate(paths, 1):
        loaded, payload = finalize_final_check_strata_common._extension_data_payload(path)
        weights, variances, edges = finalize_final_check_strata_common._comparison_current_cube(payload, path, config)
        for name in leaves:
            plane = finalize_final_check_strata__project_data_leaf(weights, variances, edges, specs[name], config)
            data[name] = finalize_final_check_strata_common.add_candidate(data[name], plane, 'Total')
        manifest.append(finalize_final_check_strata_common.candidate_input_record(path, 'Strata approved explicit VR DATA leaves only'))
        del loaded, payload, weights, variances
        finalize_final_check_strata_gc.collect()
        if index % 20 == 0 or index == len(paths):
            print(f'09 strata approved VR DATA {index}/{len(paths)}', flush=True)
    for name in allowed:
        if name in data:
            continue
        spec = specs[name]
        children = [key for key in leaves if specs[key]['plane'] == spec['plane']
                    and specs[key]['mass_low'] >= spec['mass_low']
                    and specs[key]['mass_high'] <= spec['mass_high']
                    and (spec['prompt_group'] == 'all' or specs[key]['prompt_group'] == spec['prompt_group'])]
        if not children:
            raise AssertionError('Every derived DATA stratum must be composed of approved finest leaves.')
        data[name] = finalize_final_check_strata__sum_planes(data[key] for key in children)
    if set(data) != set(allowed):
        raise AssertionError('Projected DATA selections differ from the complete-family release gates.')
    return data, manifest, leaves


def finalize_final_check_strata__validate_data_parents(data, specs, gates, base_result):
    rows = []
    cached = base_result['tables']['mass_closure']
    fields = list('ABCD') + ['var_' + region for region in 'ABCD']
    for family in gates.loc[gates.display_allowed, 'family'].unique():
        mass_slice, state = family.split('_')
        parent = finalize_final_check_strata__sum_planes(data[name] for name in data if specs[name]['family'] == family
                             and specs[name]['level'] == 'fine' and specs[name]['prompt_group'] == 'all')
        expected = cached.loc[cached.source.eq('DATA') & cached.plane.eq(state)
                              & cached.mass_slice.eq(mass_slice)]
        if len(expected) != 3 or not expected.display_allowed.all():
            raise AssertionError('The approved fine DATA parent must already be visible at every fixed WP.')
        for row in expected.itertuples():
            actual = finalize_final_check_strata_common.plane_statistics(parent, row.tx, row.ty)
            finalize_final_check_strata_np.testing.assert_allclose([actual[field] for field in fields],
                                       [getattr(row, field) for field in fields], rtol=1e-10, atol=1e-8)
        rows.append(dict(scope=family, check='Approved fine DATA sumw/sumw2 reproduce the existing coarse parent at all fixed WPs',
                         passed=True))
    return rows


def finalize_final_check_strata_finalize_stratified_data(config, result, base_result):
    """Fill only pre-approved DATA rows in a prepared MC result, in place."""
    config = finalize_final_check_strata__settings(config)
    if result['metadata']['data_loaded']:
        raise ValueError('Stratified DATA have already been loaded for this result.')
    specs, gates = result['specs'], result['tables']['strata_gate_audit']
    # Recalculate the release decisions from MC immediately before any DATA read.
    _, checked, _ = finalize_final_check_strata__purity_tables(result['planes'], specs)
    finalize_final_check_strata_pd.testing.assert_frame_equal(gates, checked)
    data, manifest, leaves = finalize_final_check_strata__load_approved_data(config, specs, gates)
    validations = finalize_final_check_strata__validate_data_parents(data, specs, gates, base_result)
    table = result['tables']['strata_closure']
    for gate in gates.loc[gates.display_allowed].to_dict('records'):
        values = finalize_final_check_strata__statistics(data[gate['selection_id']], gate['tx'], gate['ty'], config.get('min_neff', 10.))
        selected = table.source.eq('DATA') & table.selection_id.eq(gate['selection_id']) & table.wp.eq(gate['wp'])
        if selected.sum() != 1:
            raise AssertionError('Expected one DATA row for each approved stratum and fixed WP.')
        for field, value in values.items():
            table.loc[selected, field] = value
        table.loc[selected, 'display_allowed'] = True
        table.loc[selected, 'reason'] = gate['reason']
    result['planes']['data'] = data
    result['tables']['strata_input_manifest'] = finalize_final_check_strata_pd.concat(
        [result['tables']['strata_input_manifest'], finalize_final_check_strata_pd.DataFrame(manifest)], ignore_index=True)
    validations.append(dict(scope='DATA', check='Only complete-family-approved finest PP/PD/DP leaves projected; nominal/DD/high mass never projected',
                            passed=True))
    result['tables']['strata_validation'] = finalize_final_check_strata_pd.concat(
        [result['tables']['strata_validation'], finalize_final_check_strata_pd.DataFrame(validations)], ignore_index=True)
    result['metadata'].update(
        original_data_files_opened=len(manifest), data_loaded=True,
        nominal_data_evaluated=False, nominal_data_projected=False, high_mass_data_projected=False,
        data_projected_leaf_ids=leaves, data_derived_selection_ids=sorted(set(data) - set(leaves)),
        data_visible_families=sorted(gates.loc[gates.display_allowed, 'family'].unique()),
        all_fixed_wps_required_for_data=True,
        data_projection_policy='Project only finest leaves after the full 120-model × ABCD × three-WP family gate; derive approved parents by addition.')
    finalize_final_check_strata__validate_output(result)
    return result


# ========================================================================
# abcd_iso_focus: original study API
# ========================================================================
"""Isolation-capped 2mu2e study at native angle boundaries in [2.4, 2.8].

Nominal candidate selection uses absolute MC significance and regional purity.
The historical union of candidate A regions is not an optimization constraint.
VR DATA reuse the original public 01/03/07 grids and frozen report. This module
does not open original DATA; explicitly evaluated final B/C/D counts can be
attached separately, while the selected nominal A remains blinded.
"""
from pathlib import Path as finalize_iso_focus_Path
import hashlib as finalize_iso_focus_hashlib
import json as finalize_iso_focus_json

import numpy as finalize_iso_focus_np
import pandas as finalize_iso_focus_pd

finalize_iso_focus_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_iso_focus_review = _FINALIZE_NAMESPACES['review_mc_sr_working_points']
finalize_iso_focus_PROTECTED = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].PROTECTED; finalize_iso_focus__coordinate_name = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff']._coordinate_name; finalize_iso_focus__overlap = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff']._overlap; finalize_iso_focus__quantiles = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff']._quantiles; finalize_iso_focus_region_rectangles = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].region_rectangles; finalize_iso_focus_statistics_from_regions = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].statistics_from_regions
finalize_iso_focus_MEASUREMENT_FIELDS = _FINALIZE_NAMESPACES['abcd_sideband_validation'].MEASUREMENT_FIELDS; finalize_iso_focus__load_caches = _FINALIZE_NAMESPACES['abcd_sideband_validation']._load_caches


finalize_iso_focus_ANGLES = (2.4, 2.6, 2.7, 2.8)
finalize_iso_focus_REGIONS = tuple('ABCD')
finalize_iso_focus_SCOPES = [('VR1_DD', 'low', 'DD'), ('VR1_PP', 'low', 'PP'),
          ('VR1_PD', 'low', 'PD'), ('VR1_DP', 'low', 'DP'),
          ('VR2_PP', 'intermediate', 'PP'), ('VR2_PD', 'intermediate', 'PD'),
          ('VR2_DP', 'intermediate', 'DP'), ('INCLUSIVE_PP', 'all', 'PP'),
          ('INCLUSIVE_PD', 'all', 'PD'), ('INCLUSIVE_DP', 'all', 'DP')]


def finalize_iso_focus_build_iso_focus(config, previous_tables):
    """Build 20 MC candidates and reuse only previously released DATA counts.

    ``previous_tables`` must be the frozen, executed 09 scan report.  Its path
    defaults to ``09_20261027_iso_reference`` next to the current notebook and
    can be supplied as ``iso_focus_reference_dir``.  All source hashes and the
    supplied release tables are checked before any published count is reused.
    """
    if (config['topology'], config['purity_limit'], config['partner_iso_low'],
            config['partner_iso_high']) != ('2mu2e', .1, 0., .5):
        raise ValueError('Require the existing 2mu2e, partner [0,.5), and strict 10% conventions.')
    inputs, validation = [], []
    finalize_iso_focus__validate_previous(config, previous_tables, inputs, validation)
    folder = finalize_iso_focus_Path(config['vr2_output_dir'])
    background = finalize_iso_focus__read(folder/'sr_reference_grid.csv', inputs, 'Nominal background MC only')
    signals = finalize_iso_focus__read(folder/'sr_reference_signal_points.csv', inputs, 'Nominal 120 separate signal hypotheses')
    settings = finalize_iso_focus_json.loads((folder/'manifest.json').read_text())['settings']
    if (settings['topology'], settings['current_hist'], settings['expected_current_channel'],
            settings['partner_iso_low'], settings['partner_iso_high'], settings['purity_limit']) != (
            '2mu2e', config['current_hist'], config['expected_current_channel'], 0., .5, .1):
        raise AssertionError('The nominal cache settings differ from the current selection.')
    inputs.append(finalize_iso_focus_common.candidate_input_record(folder/'manifest.json', 'Nominal MC cache conventions'))
    sensitivity = finalize_iso_focus_review.signal_significance('2mu2e', background, signals)
    nominal_values = signals[['S_'+r for r in finalize_iso_focus_REGIONS] + ['var_S_'+r for r in finalize_iso_focus_REGIONS]].to_numpy(float)
    if not finalize_iso_focus_np.isfinite(nominal_values).all() or (nominal_values < 0).any():
        raise AssertionError('All nominal signal yields and variances must be finite and nonnegative.')
    bindex = background.set_index(['tx', 'ty'])
    wp2 = bindex.loc[(.2, 2.7)]
    zref = sensitivity.loc[finalize_iso_focus_np.isclose(sensitivity.tx, .2) & finalize_iso_focus_np.isclose(sensitivity.ty, 2.7)].set_index('sample')
    sref = signals.loc[finalize_iso_focus_np.isclose(signals.tx, .2) & finalize_iso_focus_np.isclose(signals.ty, 2.7)].set_index('sample')
    iso = sorted(float(x) for x in background.tx.unique() if .2 <= x <= .5)
    if len(iso) != 5 or not set(finalize_iso_focus_ANGLES).issubset(set(background.ty)):
        raise AssertionError('Require five native isolation thresholds in [.2,.5] and native angles in [2.4,2.8].')
    historical_protected = sorted(set(finalize_iso_focus_PROTECTED) | {(.3, 2.4)})
    prior_sidebands = previous_tables['sbd_regions'].loc[previous_tables['sbd_regions'].display_allowed]
    if not set(prior_sidebands.region).issubset(set('BCD')) or prior_sidebands.nominal_A_evaluated.any():
        raise AssertionError('The nominal prior must contain B/C/D only.')
    public_rectangles = [finalize_iso_focus_region_rectangles(row)[row['region']] for row in prior_sidebands.to_dict('records')]
    scan, regions, points = [], [], []
    for angle in finalize_iso_focus_ANGLES:
        for index, isolation in enumerate(iso):
            identity = dict(candidate_id=f'I_i{finalize_iso_focus__coordinate_name(isolation)}_p{finalize_iso_focus__coordinate_name(angle)}',
                source_candidate_id=f'L_i{finalize_iso_focus__coordinate_name(isolation)}_p{finalize_iso_focus__coordinate_name(angle)}',
                strategy='iso_focus', a_iso=isolation, a_phi=angle, b_iso=isolation, c_phi=angle,
                display_label=f'Isolation {isolation:g}, |dphi| {angle:g}', iso_native_step=index)
            bg = bindex.loc[(isolation, angle)]
            stat = finalize_iso_focus_statistics_from_regions({r: bg[r] for r in finalize_iso_focus_REGIONS},
                                          {r: bg['var_'+r] for r in finalize_iso_focus_REGIONS})
            point = sensitivity.loc[finalize_iso_focus_np.isclose(sensitivity.tx, isolation) & finalize_iso_focus_np.isclose(sensitivity.ty, angle)].copy()
            if len(point) != 120 or not point['sample'].is_unique or point.matching_decay.sum() != 60:
                raise AssertionError('Require all 120 hypotheses and 60 matching sensitivity models per point.')
            for metric in ('z_known_b', 'z_mc_stat_diagnostic', 's_sqrt_s_plus_b'):
                point[metric+'_ratio_WP2'] = finalize_iso_focus_review.finite_ratio(point[metric], point['sample'].map(zref[metric]))
            point['signal_retention_WP2'] = finalize_iso_focus_review.finite_ratio(point.S_A, point['sample'].map(sref.S_A))
            rectangles = finalize_iso_focus_region_rectangles(identity)
            regional = []
            for region in finalize_iso_focus_REGIONS:
                b, vb = float(bg[region]), float(bg['var_'+region])
                s, vs = point['S_'+region].to_numpy(float), point['var_S_'+region].to_numpy(float)
                good = bool(b > 1e-8 and finalize_iso_focus_np.isfinite([b, vb]).all() and vb >= 0)
                ratio = s/b if good else finalize_iso_focus_np.full(120, finalize_iso_focus_np.nan)
                worst = int(finalize_iso_focus_np.argmax(s))
                maximum = float(ratio[worst]) if good else finalize_iso_focus_np.nan
                overlap = finalize_iso_focus__overlap(rectangles[region], rectangles['A'])
                historical_overlap = any(finalize_iso_focus__overlap(rectangles[region], (0., x, y, finalize_iso_focus_np.inf))
                                         for x, y in historical_protected)
                row = dict(identity, region=region, background=b, var_background=vb,
                    background_error=finalize_iso_focus_np.sqrt(vb), neff_background=stat['neff_'+region],
                    max_S_B=maximum,
                    max_S_B_error=float(finalize_iso_focus_np.sqrt(vs[worst]/b**2+s[worst]**2*vb/b**4)) if good else finalize_iso_focus_np.nan,
                    worst_sample=point.iloc[worst]['sample'], signal_hypotheses=120,
                    valid_background=good, purity_pass=bool(good and finalize_iso_focus_np.isfinite(ratio).all() and (ratio < .1).all()),
                    overlap_protected_A=overlap, overlap_historical_A=historical_overlap,
                    nominal_data_evaluated=False)
                regions.append(row)
                regional.append(row)
                point['S_B_'+region] = ratio
            bcd = [r for r in regional if r['region'] in 'BCD']
            valid = all(r['valid_background'] for r in bcd)
            worst = max(bcd, key=lambda r:r['max_S_B']) if valid else None
            pure = all(r['purity_pass'] for r in bcd)
            safe = not any(r['overlap_protected_A'] for r in bcd)
            previous_nominal = finalize_iso_focus__published_nominal_match(identity, previous_tables)
            reusable = bool(pure and safe and previous_nominal)
            row = dict(identity, **stat, source='MC', signal_hypotheses=120,
                max_S_B_BCD=worst['max_S_B'] if worst else finalize_iso_focus_np.nan,
                worst_BCD_region=worst['region'] if worst else '',
                worst_BCD_sample=worst['worst_sample'] if worst else '',
                purity_pass=pure, geometry_safe=safe, operational_pass=bool(pure and safe),
                positive_BCD=valid, nominal_data_evaluated=False, nominal_data_permission=reusable,
                nominal_data_reuse_available=reusable, previous_nominal_candidate_id=previous_nominal,
                A_overlaps_previously_released_nominal_BCD=any(finalize_iso_focus__overlap(rectangles['A'], r) for r in public_rectangles),
                background_retention_WP2=float(bg.A/wp2.A),
                prediction_relative_error=stat['pred_error']/stat['pred'] if stat['pred'] > 0 else finalize_iso_focus_np.nan,
                gate_reason='PREVIOUSLY_RELEASED_IDENTICAL_NOMINAL_BCD_ONLY' if reusable else
                            'PURITY_PASS_FINAL_DATA_NOT_YET_EVALUATED' if pure and safe else
                            'OWN_A_OVERLAP' if pure else 'BCD_CONTAMINATION_AND_OWN_A_OVERLAP' if not safe else
                            'BCD_CONTAMINATION_AT_OR_ABOVE_10_PERCENT')
            row.update({'max_S_B_'+r['region']: r['max_S_B'] for r in bcd})
            matching = point.loc[point.matching_decay]
            for field in ('signal_retention_WP2','z_known_b','z_mc_stat_diagnostic','s_sqrt_s_plus_b',
                          'z_known_b_ratio_WP2','z_mc_stat_diagnostic_ratio_WP2','s_sqrt_s_plus_b_ratio_WP2'):
                finalize_iso_focus__quantiles(row, field, matching[field])
            point['candidate_id'] = identity['candidate_id']
            point['strategy'] = identity['strategy']
            points.append(point)
            scan.append(row)
            finalize_iso_focus_np.testing.assert_allclose([stat[r] for r in finalize_iso_focus_REGIONS], [bg[r] for r in finalize_iso_focus_REGIONS], rtol=1e-12)
            validation.append(dict(scope=identity['candidate_id'],
                check='Five native isolation thresholds at each available angle use all 120 signals and exact nominal MC regions', passed=True))
    scan = finalize_iso_focus_pd.DataFrame(scan)
    selection = finalize_iso_focus__select(scan)
    if len(scan) != 20 or not scan.geometry_safe.all():
        raise AssertionError('Every ordinary B/C/D must be disjoint from its own A.')
    if set(scan.loc[scan.nominal_data_reuse_available,'candidate_id']) != {'I_i0p3_p2p4'}:
        raise AssertionError('Only the identical previously published ordinary definition may reuse nominal controls.')
    selected = selection.loc[selection.role.eq('global_best')].iloc[0]
    if bool(selected.A_overlaps_previously_released_nominal_BCD):
        raise AssertionError('The selected final A must not overlap an already released nominal control.')
    protected = [(float(selected.a_iso), float(selected.a_phi))]
    validation.append(dict(scope='Geometry',
        check='Historical A union is not a ranking constraint; selected final A is disjoint from previously released nominal controls', passed=True))
    vr, vr_purity = finalize_iso_focus__vr_tables(config, scan, previous_tables, inputs, validation)
    signal_points = finalize_iso_focus_pd.concat(points, ignore_index=True)
    data_contamination, data_signal_points = finalize_iso_focus__nominal_data_contamination(
        selection, finalize_iso_focus_pd.DataFrame(regions), signal_points, previous_tables, validation)
    hidden = vr.loc[vr.source.eq('DATA') & ~vr.display_allowed]
    if not hidden[finalize_iso_focus_MEASUREMENT_FIELDS].isna().all().all():
        raise AssertionError('Unreleased VR statistics must remain NaN.')
    validation.append(dict(scope='DATA reuse',
        check='Only previously released identical VR rectangles are copied; all other DATA measurements remain NaN', passed=True))
    return dict(tables=dict(if_scan=scan, if_regions=finalize_iso_focus_pd.DataFrame(regions),
        if_signals=signal_points, if_selection=selection,
        if_vr_closure=vr, if_vr_purity=vr_purity,
        if_nominal_data_contamination=data_contamination,
        if_nominal_data_signal_points=data_signal_points,
        if_validation=finalize_iso_focus_pd.DataFrame(validation), if_input_manifest=finalize_iso_focus_pd.DataFrame(inputs)),
        metadata=dict(angles=list(finalize_iso_focus_ANGLES), isolation_thresholds=iso, signal_hypotheses=120,
            sensitivity_hypotheses=60, candidates=20, isolation_boundary_cap=.5,
            requested_angle_range=[2.4,2.8], unevaluated_non_native_angles=[2.5],
            primary_objective='Maximum absolute median known-background MC-A Z over 60 matching hypotheses after all-120 B/C/D purity; isolation<=.5, native dphi in [2.4,2.8]. No historical-A optimization veto.',
            additional_sensitivity_metric='s_sqrt_s_plus_b=S_A_MC/sqrt(S_A_MC+B_A_MC); absolute and paired WP2-ratio quantiles over the same 60 matching hypotheses. Diagnostic only; Z remains the primary ranking.',
            main_candidate_id=selection.loc[selection.role.eq('global_best'),'candidate_id'].iloc[0],
            finite_mc_candidate_id=selection.loc[selection.role.eq('global_finite_check'),'candidate_id'].iloc[0],
            operational_candidate_id=selection.loc[selection.role.eq('global_operational_best'),'candidate_id'].iloc[0],
            selected_loose_candidate_id=selection.loc[selection.role.eq('global_operational_best'),'candidate_id'].iloc[0],
            protected_boundaries=protected,
            protected_union=f'isolation<{selected.a_iso:g} and |dphi|>={selected.a_phi:g}; selected final loose A only, containing WP2 A.',
            historical_A_union_used_for_selection=False,
            nominal_data_evaluated=False, new_vr_data_evaluated=False, original_data_files_opened=0,
            nominal_A_evaluated=False, new_nominal_BCD_counts_evaluated=False,
            nominal_data_diagnostic_evaluated=True, nominal_BCD_counts_reused=True,
            reused_vr_data_rows=int(vr.loc[vr.source.eq('DATA'),'display_allowed'].sum()),
            reused_vr_data_by_source=vr.loc[vr.source.eq('DATA')&vr.display_allowed,'data_reuse_source'].value_counts().to_dict(),
            previously_public_rows_missing_from_09_subset=int(vr.loc[vr.source.eq('DATA'),'prior_subset_omitted_public_measurement'].sum()),
            nominal_data_policy='MC chooses the final point without historical-A veto. Existing controls are reused here; the separate final-controls evaluation checks all proposed B/C/D against the selected final A before reading DATA.',
            nominal_data_diagnostic='For previously released controls only, S_MC/N_DATA is a data-count signal-fraction proxy, not S/B. S_MC/(N_DATA-S_MC) is a per-hypothesis mu=1 subtraction diagnostic, not a fitted background. No DATA ratio changes the MC selection or blinding rule.',
            vr_data_policy='Reuse exact already public native-grid measurements from 01/03/07 as well as the frozen preceding 09 release. Current all-120 purity and source identity are checked; no new integrated DATA release.',
            prior_release_policy_note='The preceding 261-row release was an incomplete record of older public integrals. Exact 01/03/07 observations remain public even when that subset audit withheld them; redisplaying the same query adds no information.',
            inclusive_signal_policy='Use the validated notebook-07 all-120 regional contamination envelope and MC statistics; never sum maxima across mass slices.',
            interpretation='Isolation boundaries are capped at .5; angles use only exact native edges in [2.4,2.8], with no interpolation at 2.5. Scores are diagnostics, not final-likelihood optimality.'))


def finalize_iso_focus__read(path, inputs, role):
    inputs.append(finalize_iso_focus_common.candidate_input_record(path, role))
    return finalize_iso_focus_pd.read_csv(path, float_precision='round_trip')


def finalize_iso_focus__validate_previous(config, supplied, inputs, validation):
    directory = finalize_iso_focus_Path(config.get('iso_focus_reference_dir', finalize_iso_focus_Path(config['notebook_path']).parent/'09_20261027_iso_reference'))
    path = directory/'archive_manifest.json'
    document = finalize_iso_focus_json.loads(path.read_text())
    for relative, expected in document['files'].items():
        if finalize_iso_focus_hashlib.sha256((directory/relative).read_bytes()).hexdigest() != expected:
            raise AssertionError('The frozen preceding scan reference changed: '+relative)
    inputs.append(finalize_iso_focus_common.candidate_input_record(path,'Frozen preceding report: published VR and nominal-sideband masks'))
    for name in ('sv_closure','sv_release_audit','sbd_regions','sbd_summary','sideband_signals','sideband_regions'):
        filename=directory/'09_2mu2e_final_check_outputs'/(name+'.csv')
        expected=finalize_iso_focus__read(filename,inputs,'Previously published '+name)
        finalize_iso_focus_pd.testing.assert_frame_equal(supplied[name].reset_index(drop=True),expected,check_dtype=False)
    if supplied['sv_closure'].query("source == 'DATA'").display_allowed.sum() != 261:
        raise AssertionError('Require the established 261 published VR rows as the frozen release set.')
    validation.append(dict(scope='Previous report',check='Frozen report hashes and supplied previously released measurements agree',passed=True))


def finalize_iso_focus__select(scan):
    rows=[]
    primary='z_known_b_median'
    for angle in finalize_iso_focus_ANGLES:
        accepted=scan.loc[finalize_iso_focus_np.isclose(scan.a_phi,angle) & scan.purity_pass]
        if accepted.empty:
            rows.append(dict(candidate_id='',source_candidate_id='',strategy='iso_focus',a_phi=angle,
                a_iso=finalize_iso_focus_np.nan,b_iso=finalize_iso_focus_np.nan,c_phi=angle,display_label=f'No purity pass at |dphi|={angle:g}',
                role='angle_no_purity_pass',selection_status='NO_PURITY_PASS_WITHIN_ISO_CAP',
                objective='No evaluated isolation boundary<=.5 passes all-120 B/C/D purity at this angle',
                purity_pass=False,geometry_safe=False,operational_pass=False,nominal_data_permission=False))
            continue
        first=accepted.sort_values('a_iso').iloc[0]
        best=accepted.sort_values([primary,'a_iso'],ascending=[False,True]).iloc[0]
        for role,row,objective in [('angle_first_pass',first,'Smallest isolation boundary passing all-120 B/C/D purity at this fixed angle'),
                                   ('angle_best',best,'Largest primary sensitivity score among purity-passing isolation cuts at this fixed angle')]:
            rows.append(dict(row,role=role,objective=objective,selection_status='MC_PURITY_PASS'))
    accepted=scan.loc[scan.purity_pass]
    for role,metric in [('global_best',primary),('global_finite_check','z_mc_stat_diagnostic_median')]:
        best=accepted.sort_values([metric,'a_iso','a_phi'],ascending=[False,True,True]).iloc[0]
        rows.append(dict(best,role=role,objective='Maximum '+metric+' within the capped native grid',
                         selection_status='MC_PURITY_PASS'))
    eligible=scan.loc[scan.operational_pass]
    if eligible.empty:
        raise AssertionError('The capped scan must retain at least one purity-passing ordinary candidate.')
    best=eligible.sort_values([primary,'a_iso','a_phi'],ascending=[False,True,True]).iloc[0]
    rows.append(dict(best,role='global_operational_best',selection_status='FINAL_MC_SELECTED',
        objective='Compatibility alias of the largest absolute median MC significance after B/C/D purity; historical A union is not a selection constraint'))
    return finalize_iso_focus_pd.DataFrame(rows)


def finalize_iso_focus__vr_tables(config, scan, previous_tables, inputs, validation):
    caches=finalize_iso_focus__load_caches(config,inputs,validation)
    folder=finalize_iso_focus_Path(config['comprehensive_dir'])
    inclusive_grid=finalize_iso_focus__read(folder/'mass_grid.csv',inputs,'Inclusive VR MC and already published native-grid DATA')
    inclusive=inclusive_grid.query("source == 'MC'")
    inclusive_purity=finalize_iso_focus__read(folder/'mass_purity_grid_regions.csv',inputs,'Validated all-120 inclusive regional contamination maxima')
    if len(inclusive)!=462 or not inclusive.signal_hypotheses.eq(120).all():
        raise AssertionError('Inclusive MC must cover 154 native coordinates in PP/PD/DP with all 120 signals.')
    published=previous_tables['sv_closure'].query("source == 'DATA' and display_allowed")
    release_history=previous_tables['sv_release_audit']
    public_grids=finalize_iso_focus__validate_public_vr_grids(config,caches,inclusive_grid,inputs,validation)
    closure,purity=[],[]
    for scope,mass,state in finalize_iso_focus_SCOPES:
        grid=inclusive.loc[inclusive.plane.eq(state)] if mass=='all' else caches[mass]['grid'].query("source == 'MC' and plane == @state")
        for spec in scan.to_dict('records'):
            identity={k:spec[k] for k in ('candidate_id','source_candidate_id','strategy','a_iso','a_phi','b_iso','c_phi','display_label')}
            identity.update(scope=scope,plane=state,mass_slice=mass)
            row=grid.loc[finalize_iso_focus_np.isclose(grid.tx,spec['a_iso']) & finalize_iso_focus_np.isclose(grid.ty,spec['a_phi'])]
            if len(row)!=1:
                raise AssertionError('Every focused VR MC coordinate must be unique.')
            bg=row.iloc[0]
            stat=finalize_iso_focus_statistics_from_regions({r:bg[r] for r in finalize_iso_focus_REGIONS},{r:bg['var_'+r] for r in finalize_iso_focus_REGIONS})
            if mass!='all':
                sig=caches[mass]['signals']
                sig=sig.loc[sig.plane.eq(state) & finalize_iso_focus_np.isclose(sig.tx,spec['a_iso']) & finalize_iso_focus_np.isclose(sig.ty,spec['a_phi'])]
                if len(sig)!=120 or not sig['sample'].is_unique:
                    raise AssertionError('Each focused low/intermediate VR must contain 120 separate signals.')
            gates=[]
            for region in finalize_iso_focus_REGIONS:
                b=float(bg[region])
                if mass=='all':
                    cached=inclusive_purity.loc[inclusive_purity.plane.eq(state)&inclusive_purity.region.eq(region)
                        & finalize_iso_focus_np.isclose(inclusive_purity.tx,spec['a_iso']) & finalize_iso_focus_np.isclose(inclusive_purity.ty,spec['a_phi'])]
                    if len(cached)!=1:raise AssertionError('Inclusive regional maxima must be unique.')
                    maximum=float(cached.iloc[0].max_S_B)
                    sample=cached.iloc[0].worst_sample
                    good=bool(b>1e-8 and finalize_iso_focus_np.isfinite(maximum) and maximum>=0)
                else:
                    s=sig['S_'+region].to_numpy(float)
                    if not finalize_iso_focus_np.isfinite(s).all() or (s<0).any():raise AssertionError('Invalid VR signal yields.')
                    worst=int(finalize_iso_focus_np.argmax(s))
                    maximum=float(s[worst]/b) if b>1e-8 else finalize_iso_focus_np.nan
                    sample=sig.iloc[worst]['sample']
                    good=bool(b>1e-8 and finalize_iso_focus_np.isfinite(maximum))
                gate=dict(identity,region=region,background=b,var_background=float(bg['var_'+region]),
                    max_S_B=maximum,worst_sample=sample,signal_hypotheses=120,
                    valid_background=good,approved=bool(good and maximum<.1))
                gates.append(gate);purity.append(gate)
            approved=all(r['approved'] for r in gates)
            maximum=max(r['max_S_B'] for r in gates) if all(finalize_iso_focus_np.isfinite(r['max_S_B']) for r in gates) else finalize_iso_focus_np.nan
            closure.append(dict(identity,**stat,source='MC',approved=approved,max_S_B=maximum,
                signal_hypotheses=120,display_allowed=True,reason='MC_ALWAYS_VISIBLE',new_data_evaluated=False,
                reused_previous_data=False,previous_candidate_id=''))
            previous=published.loc[published.scope.eq(scope)&finalize_iso_focus_np.isclose(published.a_iso,spec['a_iso'])
                &finalize_iso_focus_np.isclose(published.a_phi,spec['a_phi'])&finalize_iso_focus_np.isclose(published.b_iso,spec['b_iso'])
                &finalize_iso_focus_np.isclose(published.c_phi,spec['c_phi'])]
            original=public_grids[mass]
            original=original.loc[original.source.eq('DATA')&original.plane.eq(state)
                &finalize_iso_focus_np.isclose(original.tx,spec['a_iso'],rtol=0,atol=1e-12)
                &finalize_iso_focus_np.isclose(original.ty,spec['a_phi'],rtol=0,atol=1e-12)&original.approved]
            if len(original)>1:
                raise AssertionError('An original public native grid must have one unique exact geometry.')
            if len(original):
                if not approved:
                    raise AssertionError('Previously public identical geometry no longer agrees with the current all-120 purity gate.')
                old_mc=public_grids[mass].loc[public_grids[mass].source.eq('MC')&public_grids[mass].plane.eq(state)
                    &finalize_iso_focus_np.isclose(public_grids[mass].tx,spec['a_iso'],rtol=0,atol=1e-12)
                    &finalize_iso_focus_np.isclose(public_grids[mass].ty,spec['a_phi'],rtol=0,atol=1e-12)]
                if len(old_mc)!=1:raise AssertionError('Public DATA requires its exact unique original MC counterpart.')
                finalize_iso_focus_np.testing.assert_allclose([old_mc.iloc[0][f] for r in finalize_iso_focus_REGIONS for f in (r,'var_'+r)],
                    [stat[f] for r in finalize_iso_focus_REGIONS for f in (r,'var_'+r)],rtol=1e-10,atol=1e-10)
                finalize_iso_focus_np.testing.assert_allclose(original.iloc[0].max_S_B,maximum,rtol=1e-8,atol=1e-10)
                for copied in previous.to_dict('records'):
                    finalize_iso_focus_np.testing.assert_allclose([copied[f] for f in finalize_iso_focus_MEASUREMENT_FIELDS],
                        [original.iloc[0][f] for f in finalize_iso_focus_MEASUREMENT_FIELDS],rtol=1e-10,atol=1e-10,equal_nan=True)
            blank={field:finalize_iso_focus_np.nan for field in finalize_iso_focus_MEASUREMENT_FIELDS}
            blank.update(status='BLINDED',kappa_status='BLINDED',low_stat=False)
            show=bool(approved and (len(previous) or len(original)))
            reason=('PREVIOUSLY_RELEASED_IDENTICAL_VR' if show else
                    'VR_CONTAMINATION_OR_INVALID_BACKGROUND' if not approved else
                    'NO_PREVIOUSLY_RELEASED_IDENTICAL_VR')
            release=release_history.loc[release_history.scope.eq(scope)]
            for field in ('a_iso','a_phi','b_iso','c_phi'):
                release=release.loc[finalize_iso_focus_np.isclose(release[field],spec[field],rtol=0,atol=1e-12)]
            exact_id=release.loc[release.candidate_id.eq(spec['source_candidate_id'])]
            if len(exact_id):
                release=exact_id
            prior_release_id=''
            prior_release_reason=''
            if len(release):
                prior_release_id=str(release.iloc[0].candidate_id)
                prior_release_reason=str(release.iloc[0].reason)
                if approved and not show:
                    if release.display_allowed.any():
                        raise AssertionError('An approved prior release must have its matching published DATA measurement.')
                    reason=prior_release_reason
            old_id=''
            reuse_source=''
            missing_from_subset=bool(show and not len(previous) and len(original))
            if show:
                chosen=previous.iloc[0] if len(previous) else original.iloc[0]
                if not chosen.approved or chosen.signal_hypotheses!=120:
                    raise AssertionError('A reused VR measurement must retain its original all-120 approval.')
                for other in previous.iloc[1:].itertuples():
                    finalize_iso_focus_np.testing.assert_allclose([getattr(other,f) for f in finalize_iso_focus_MEASUREMENT_FIELDS],
                        [chosen[f] for f in finalize_iso_focus_MEASUREMENT_FIELDS],rtol=1e-10,atol=1e-10,equal_nan=True)
                blank={field:chosen[field] for field in finalize_iso_focus_MEASUREMENT_FIELDS+['status','kappa_status','low_stat']}
                if len(previous):
                    old_id=chosen.candidate_id
                    reuse_source='frozen_09_scan'
                else:
                    notebook={'low':'01','intermediate':'03','all':'07'}[mass]
                    old_id=f'{notebook}:{state}:I={spec["a_iso"]:g}:phi={spec["a_phi"]:g}'
                    reuse_source=notebook+'_public_native_grid'
                    reason='PREVIOUSLY_PUBLIC_NATIVE_GRID_'+notebook+'_EXACT_REUSE'
                finalize_iso_focus_np.testing.assert_allclose(maximum,chosen.max_S_B,rtol=1e-8,atol=1e-10)
            closure.append(dict(identity,**blank,source='DATA',approved=approved,max_S_B=maximum,
                signal_hypotheses=120,display_allowed=show,reason=reason,new_data_evaluated=False,
                reused_previous_data=show,previous_candidate_id=old_id,
                previous_release_candidate_id=prior_release_id,previous_release_reason=prior_release_reason,
                data_reuse_source=reuse_source,prior_subset_omitted_public_measurement=missing_from_subset))
        validation.append(dict(scope=scope,check='20 focused MC definitions use native cache moments and all-120 regional gates; DATA reuse requires exact previously published geometry',passed=True))
    return finalize_iso_focus_pd.DataFrame(closure),finalize_iso_focus_pd.DataFrame(purity)


def finalize_iso_focus__nominal_data_contamination(selection, focused_regions, focused_signals, previous, validation):
    """Compare signal MC with supplied, individually evaluated B/C/D counts.

    The observed count contains any signal as well as background.  S/N is
    therefore labelled as a count-fraction proxy.  The optional S/(N-S) uses
    each stored hypothesis at mu=1 separately and is never a background fit.
    """
    approved=previous['sbd_summary'].loc[previous['sbd_summary'].display_allowed].copy()
    expected={'G_i0p3_p2','L_i0p3_p2p4','G_i2_p2'}
    if not expected.issubset(set(approved.candidate_id)) or approved.nominal_A_evaluated.any():
        raise AssertionError('Retain the three previous nominal B/C/D definitions and never an A measurement.')
    focused=selection.loc[selection.role.isin(['angle_best','global_operational_best'])].drop_duplicates('candidate_id')
    current_approved=approved.loc[approved.candidate_id.ne('G_i2_p2')]
    targets=[(row,row['candidate_id'],False) for row in current_approved.to_dict('records')]
    targets += [(row,finalize_iso_focus__published_nominal_match(row,previous),True) for row in focused.to_dict('records')]
    summaries,models=[],[]
    for spec,old_id,is_focused in targets:
        available=bool(old_id)
        identity={key:spec[key] for key in ('candidate_id','strategy','a_iso','a_phi','b_iso','c_phi','display_label')}
        evaluated_now = bool(available and approved.set_index('candidate_id').loc[old_id].get('new_data_evaluated',False))
        identity.update(previous_nominal_candidate_id=old_id,
            reused_identical_nominal_geometry=available and not evaluated_now,focused_definition=is_focused)
        source_signals=focused_signals if is_focused else previous['sideband_signals']
        sig=source_signals.loc[source_signals.candidate_id.eq(spec['candidate_id'])].sort_values('sample')
        if len(sig)!=120 or not sig['sample'].is_unique:
            raise AssertionError('Nominal count-fraction diagnostics require all 120 separate signals.')
        source_regions=focused_regions if is_focused else previous['sideband_regions']
        regional=source_regions.loc[source_regions.candidate_id.eq(spec['candidate_id'])].set_index('region')
        if available and is_focused:
            if not bool(spec['geometry_safe']) or not bool(spec['purity_pass']):
                raise AssertionError('Focused DATA requires disjoint own A and the MC purity gate.')
            original=previous['sideband_signals'].loc[previous['sideband_signals'].candidate_id.eq(old_id)].sort_values('sample')
            if list(sig['sample'])!=list(original['sample']):
                raise AssertionError('An aliased nominal definition must retain the same signal hypotheses.')
            columns=['S_'+r for r in 'BCD']+['var_S_'+r for r in 'BCD']
            finalize_iso_focus_np.testing.assert_allclose(sig[columns],original[columns],rtol=1e-9,atol=1e-10)
        for region in 'BCD':
            b=float(regional.loc[region,'background'])
            vb=float(regional.loc[region,'var_background'])
            s=sig['S_'+region].to_numpy(float)
            vs=sig['var_S_'+region].to_numpy(float)
            if not finalize_iso_focus_np.isfinite(s).all() or not finalize_iso_focus_np.isfinite(vs).all() or finalize_iso_focus_np.any(s<0) or finalize_iso_focus_np.any(vs<0):
                raise AssertionError('Stored signal hypothesis yields and variances must be nonnegative.')
            mc_ratio=s/b if b>1e-8 else finalize_iso_focus_np.full(120,finalize_iso_focus_np.nan)
            maximum_index=int(finalize_iso_focus_np.argmax(s))
            maximum_mc=float(mc_ratio[maximum_index]) if b>1e-8 else finalize_iso_focus_np.nan
            expected_max=float(regional.loc[region,'max_S_B'])
            finalize_iso_focus_np.testing.assert_allclose(maximum_mc,expected_max,rtol=1e-9,atol=1e-10,equal_nan=True)
            n,vn=finalize_iso_focus_np.nan,finalize_iso_focus_np.nan
            reason=('OWN_A_OVERLAP_NO_NOMINAL_QUERY' if not bool(spec.get('geometry_safe',False)) else
                    'NOMINAL_BCD_NOT_EVALUATED')
            if available:
                source=previous['sbd_regions'].loc[previous['sbd_regions'].candidate_id.eq(old_id)
                    &previous['sbd_regions'].region.eq(region)]
                if len(source)!=1 or not source.iloc[0].display_allowed or source.iloc[0].nominal_A_evaluated:
                    raise AssertionError('Every nominal DATA diagnostic needs an individually published B/C/D count.')
                n,vn=float(source.iloc[0].value),float(source.iloc[0].variance)
                if not finalize_iso_focus_np.isfinite([n,vn]).all() or n<0 or n!=round(n) or vn!=n:
                    raise AssertionError('Published nominal controls must be unweighted DATA counts.')
                reason=('FINAL_SELECTED_BCD_EVALUATED_WITH_A_BLINDED' if evaluated_now else
                        'PREVIOUSLY_RELEASED_NOMINAL_BCD_REUSED_ONLY')
            count_valid=bool(available and n>0)
            count_ratio=s/n if count_valid else finalize_iso_focus_np.full(120,finalize_iso_focus_np.nan)
            count_error=finalize_iso_focus_np.sqrt(vs/n**2+s*s*vn/n**4) if count_valid else finalize_iso_focus_np.full(120,finalize_iso_focus_np.nan)
            subtracted=n-s if available else finalize_iso_focus_np.full(120,finalize_iso_focus_np.nan)
            subtraction_valid=available & finalize_iso_focus_np.isfinite(subtracted) & (subtracted>0)
            hypothetical=finalize_iso_focus_np.full(120,finalize_iso_focus_np.nan)
            finalize_iso_focus_np.divide(s,subtracted,out=hypothetical,where=subtraction_valid)
            all_subtracted=bool(finalize_iso_focus_np.all(subtraction_valid))
            max_hypothetical=float(finalize_iso_focus_np.max(hypothetical)) if all_subtracted else finalize_iso_focus_np.nan
            record=dict(identity,region=region,background_mc=b,var_background_mc=vb,
                signal_hypotheses=120,signal_max=float(s[maximum_index]),
                var_signal_max=float(vs[maximum_index]),worst_sample=sig.iloc[maximum_index]['sample'],
                max_S_B_MC=maximum_mc,
                max_S_B_MC_error=float(finalize_iso_focus_np.sqrt(vs[maximum_index]/b**2+s[maximum_index]**2*vb/b**4)) if b>1e-8 else finalize_iso_focus_np.nan,
                mc_purity_pass=bool(b>1e-8 and finalize_iso_focus_np.isfinite(mc_ratio).all() and (mc_ratio<.1).all()),
                data_available=available,display_allowed=available,data_count=n,
                data_count_error=finalize_iso_focus_np.sqrt(vn) if available else finalize_iso_focus_np.nan,
                max_S_over_N_data=float(count_ratio[maximum_index]) if count_valid else finalize_iso_focus_np.nan,
                max_S_over_N_data_error=float(count_error[maximum_index]) if count_valid else finalize_iso_focus_np.nan,
                data_fraction_below_10pct_all_models=bool(count_valid and finalize_iso_focus_np.all(count_ratio<.1)),
                max_S_over_N_minus_S=max_hypothetical,
                hypothetical_background_positive_all_models=all_subtracted,
                hypothetical_fraction_below_10pct_all_models=bool(all_subtracted and finalize_iso_focus_np.all(hypothetical<.1)),
                nominal_A_evaluated=False,new_data_evaluated=evaluated_now,reason=reason,
                ratio_used_for_selection_or_release=False)
            summaries.append(record)
            for i,sample in enumerate(sig['sample']):
                models.append(dict(identity,region=region,sample=sample,S_MC=float(s[i]),var_S_MC=float(vs[i]),
                    background_MC=b,var_background_MC=vb,S_B_MC=float(mc_ratio[i]),
                    data_count=n,data_count_variance=vn,S_over_N_data=float(count_ratio[i]),
                    S_over_N_data_error=float(count_error[i]),
                    N_minus_S_at_mu1=float(subtracted[i]),S_over_N_minus_S=float(hypothetical[i]),
                    hypothetical_background_positive=bool(subtraction_valid[i]),
                    data_available=available,display_allowed=available,nominal_A_evaluated=False,
                    new_data_evaluated=evaluated_now,reason=reason))
    summary,model_table=finalize_iso_focus_pd.DataFrame(summaries),finalize_iso_focus_pd.DataFrame(models)
    if (len(summary)!=len(targets)*3 or len(model_table)!=len(targets)*3*120
            or not model_table.groupby(['candidate_id','region'])['sample'].nunique().eq(120).all()):
        raise AssertionError('Each current diagnostic definition requires three sidebands and 120 separate hypotheses.')
    if summary.candidate_id.eq('G_i2_p2').any():
        raise AssertionError('The former isolation-2 stress test is historical only, outside the current capped comparison.')
    hidden=summary.loc[~summary.data_available]
    data_columns=['data_count','data_count_error','max_S_over_N_data','max_S_over_N_data_error','max_S_over_N_minus_S']
    if not hidden[data_columns].isna().all().all():
        raise AssertionError('Unevaluated nominal candidates must expose no DATA or DATA-derived ratio.')
    if not model_table.loc[~model_table.data_available,
        ['data_count','data_count_variance','S_over_N_data','S_over_N_data_error','N_minus_S_at_mu1','S_over_N_minus_S']].isna().all().all():
        raise AssertionError('Blocked model-level nominal DATA diagnostics must remain NaN.')
    validation.extend([
        dict(scope='Nominal DATA diagnostic',check='Current definitions use only supplied evaluated B/C/D counts; aliases retain identical signal yields and every unevaluated definition remains withheld',passed=True),
        dict(scope='Nominal DATA diagnostic',check='120 individual signal hypotheses per candidate and sideband; S/N and hypothetical mu=1 S/(N-S) are distinct from direct-MC S/B',passed=True),
        dict(scope='Nominal DATA diagnostic',check='Nonpositive DATA or hypothetical background denominators remain undefined; DATA diagnostics do not alter selection or release flags',passed=True),
    ])
    return summary,model_table


def finalize_iso_focus__published_nominal_match(spec, previous):
    """Return an already released ID for exactly the same four boundaries."""
    approved=previous['sbd_summary'].loc[previous['sbd_summary'].display_allowed]
    match=finalize_iso_focus_np.ones(len(approved),dtype=bool)
    for field in ('a_iso','a_phi','b_iso','c_phi'):
        match &= finalize_iso_focus_np.isclose(approved[field],spec[field],rtol=0,atol=1e-12)
    rows=approved.loc[match]
    if len(rows)>1:
        raise AssertionError('The published nominal geometry must identify one prior definition uniquely.')
    if rows.empty:
        return ''
    row=rows.iloc[0]
    if bool(row.nominal_A_evaluated):
        raise AssertionError('Previously released controls cannot contain a nominal DATA A measurement.')
    return str(row.candidate_id)


def finalize_iso_focus__validate_public_vr_grids(config,caches,inclusive,inputs,validation):
    """Verify existing 01/03/07 numeric DATA artifacts, without new DATA reads."""
    grids={'low':caches['low']['grid'],'intermediate':caches['intermediate']['grid'],'all':inclusive}
    jobs=[('low',finalize_iso_focus_Path(config['vr1_output_dir']),'01','grid',{'DD','PP','PD','DP'},(0.,150.)),
          ('intermediate',finalize_iso_focus_Path(config['vr2_output_dir']),'03','grid',{'PP','PD','DP'},(150.,300.)),
          ('all',finalize_iso_focus_Path(config['comprehensive_dir']),'07','mass_grid',{'PP','PD','DP'},None)]
    production_reference=None
    for mass,folder,notebook,stem,states,bounds in jobs:
        manifest_path=folder/'manifest.json'
        document=finalize_iso_focus_json.loads(manifest_path.read_text())
        settings=document['configuration'] if mass=='all' else document['settings']
        for key in ('topology','current_hist','expected_current_channel','partner_iso_low','partner_iso_high','purity_limit'):
            if settings[key]!=config[key]:
                raise AssertionError('Original public VR population differs: '+notebook+'/'+key)
        if bounds is not None and (settings['mass_min'],settings['mass_max'])!=bounds:
            raise AssertionError('Original public VR mass window differs.')
        if mass=='all':
            method=document['mass_methodology']
            if (set(method['states'])!={'PP','PD','DP'} or method['nominal_data_projected']
                    or method['mass_selection']!='All physical MJJ >= 0 with no upper cut; numeric mass underflow excluded, overflow retained.'):
                raise AssertionError('Original inclusive mass/state/flow selection differs.')
            if document['table_rows'].get(stem)!=len(grids[mass]):
                raise AssertionError('The public inclusive numeric grid must match its exported manifest row count.')
        frame=grids[mass]
        if (len(frame)!=len(states)*154*2 or set(frame.plane)!=states
                or frame.duplicated(['source','plane','tx','ty']).any()
                or not frame.groupby(['source','plane']).size().eq(154).all()
                or not frame.signal_hypotheses.eq(120).all()):
            raise AssertionError('Each public native grid must contain 154 MC/DATA coordinates per explicit state and all 120 signals.')
        path=folder/(stem+'.csv')
        source=finalize_iso_focus_pd.read_csv(path,float_precision='round_trip')
        finalize_iso_focus_pd.testing.assert_frame_equal(frame.reset_index(drop=True),source,check_dtype=False)
        inputs.append(finalize_iso_focus_common.candidate_input_record(path,'Previously public '+notebook+' exact native-grid DATA and MC'))
        public=frame.loc[frame.source.eq('DATA')&frame.approved]
        values=public[list(finalize_iso_focus_REGIONS)].to_numpy(float)
        variances=public[['var_'+r for r in finalize_iso_focus_REGIONS]].to_numpy(float)
        if (not finalize_iso_focus_np.isfinite(values).all() or (values<0).any()
                or not finalize_iso_focus_np.array_equal(values,finalize_iso_focus_np.rint(values)) or not finalize_iso_focus_np.array_equal(values,variances)
                or not finalize_iso_focus_np.isfinite(public.max_S_B).all() or not public.max_S_B.lt(.1).all()):
            raise AssertionError('A public DATA reuse source needs finite integer ABCD counts, variance=N and its original strict approval.')
        hidden=frame.loc[frame.source.eq('DATA')&~frame.approved]
        if not hidden[list(finalize_iso_focus_REGIONS)+['var_'+r for r in finalize_iso_focus_REGIONS]].isna().all().all():
            raise AssertionError('Original grid rows failing their DATA gate must remain masked.')
        figure_path=folder/'figure_manifest.csv'
        figures=finalize_iso_focus_pd.read_csv(figure_path)
        field='name' if mass=='all' else 'figure'
        required=['mass_scan_'+state for state in ('PP','PD','DP')] if mass=='all' else ['data_VR_closure_grid']
        recorded=finalize_iso_focus_pd.DataFrame(document['figures'])
        for name in required:
            match=figures.loc[figures[field].eq(name)]
            registry=recorded.loc[recorded[field].eq(name)]
            if len(match)!=1 or len(registry)!=1:
                raise AssertionError('Original public DATA scan/grid figures must exist in both exported figure registries.')
            for extension in ('png',):
                if match.iloc[0][extension]!=registry.iloc[0][extension]:
                    raise AssertionError('Original DATA figure manifests disagree.')
                artifact=folder/match.iloc[0][extension]
                if not artifact.is_file():raise AssertionError('Missing original public DATA figure: '+str(artifact))
                inputs.append(finalize_iso_focus_common.candidate_input_record(artifact,'Previously exported '+notebook+' DATA scan/grid figure'))
        inputs.extend([finalize_iso_focus_common.candidate_input_record(manifest_path,'Original '+notebook+' public export and selection manifest'),
                       finalize_iso_focus_common.candidate_input_record(figure_path,'Original '+notebook+' DATA figure registry')])
        production_path=folder/('mass_input_manifest.csv' if mass=='all' else 'input_manifest.csv')
        production=finalize_iso_focus_pd.read_csv(production_path)
        label='DATA mass-inclusive PP/PD/DP only' if mass=='all' else 'DATA selected VR candidates'
        production=production.loc[production.source.eq(label),['path','size_bytes','sha256']].sort_values('path').reset_index(drop=True)
        if len(production)!=68 or production.path.duplicated().any():
            raise AssertionError('Require the same 68 original DATA production records for every reused public grid.')
        if production_reference is None:
            production_reference=production
        else:
            finalize_iso_focus_pd.testing.assert_frame_equal(production,production_reference,check_dtype=False)
        inputs.append(finalize_iso_focus_common.candidate_input_record(production_path,'Original '+notebook+' DATA-production SHA256 records; no original DATA read'))
        validation.append(dict(scope='Public grid '+notebook,
            check='Existing numeric grid, displayed scan/grid artifacts, source population and identical 68-file DATA hashes justify exact-query reuse without new disclosure',passed=True))
    return grids


def finalize_iso_focus_attach_final_nominal_controls(result, previous_tables, final_controls):
    """Attach explicit final B/C/D observations without changing MC selection."""
    previous = dict(previous_tables)
    previous.update(final_controls['tables'])
    tables = result['tables']
    selected_before = tables['if_selection'][['role', 'candidate_id']].copy()
    evaluated = previous['sbd_summary'].loc[previous['sbd_summary'].display_allowed]
    for name in ('if_scan', 'if_selection'):
        frame = tables[name]
        for index, spec in frame.iterrows():
            if not spec.get('candidate_id') or not finalize_iso_focus_np.isfinite(spec.get('a_iso', finalize_iso_focus_np.nan)):
                continue
            identity = finalize_iso_focus__published_nominal_match(spec, previous)
            if not identity:
                continue
            source = evaluated.loc[evaluated.candidate_id.eq(identity)].iloc[0]
            fresh = bool(source.get('new_data_evaluated', False))
            frame.loc[index, 'previous_nominal_candidate_id'] = identity
            frame.loc[index, 'nominal_data_permission'] = True
            frame.loc[index, 'nominal_data_evaluated'] = fresh
            frame.loc[index, 'nominal_data_reuse_available'] = not fresh
            frame.loc[index, 'gate_reason'] = ('FINAL_SELECTED_BCD_EVALUATED_WITH_A_BLINDED' if fresh
                else 'PREVIOUSLY_RELEASED_IDENTICAL_NOMINAL_BCD_ONLY')
    checks = []
    diagnostic, models = finalize_iso_focus__nominal_data_contamination(tables['if_selection'], tables['if_regions'],
        tables['if_signals'], previous, checks)
    tables['if_nominal_data_contamination'] = diagnostic
    tables['if_nominal_data_signal_points'] = models
    tables['if_validation'] = finalize_iso_focus_pd.concat([tables['if_validation'], finalize_iso_focus_pd.DataFrame(checks)], ignore_index=True)
    tables['if_input_manifest'] = finalize_iso_focus_pd.concat([tables['if_input_manifest'],
        final_controls['tables']['nominal_final_input_manifest']], ignore_index=True)
    finalize_iso_focus_pd.testing.assert_frame_equal(selected_before, tables['if_selection'][['role', 'candidate_id']])
    result['metadata'].update(nominal_data_evaluated=bool(final_controls['metadata']['new_candidates']),
        new_nominal_BCD_counts_evaluated=bool(final_controls['metadata']['new_candidates']),
        original_data_files_opened=final_controls['metadata']['original_data_files_opened'],
        final_nominal_controls=final_controls['metadata'], nominal_A_evaluated=False,
        nominal_data_diagnostic='Separate S_MC/N_DATA and conditional mu=1 S_MC/(N_DATA-S_MC) diagnostics use evaluated final B/C/D only; they do not rerank MC-selected candidates.')
    return result


# ========================================================================
# abcd_sideband_data: original study API
# ========================================================================
"""Prospective MC disclosure audit and optional nominal B/C/D DATA counts.

The default performs no original DATA access. With explicit evaluation enabled,
only jointly approved B/C/D six-dimensional slices are summed. Nominal A is
never indexed, projected, counted or used to compute a DATA closure/significance.
"""
from pathlib import Path as finalize_sideband_data_Path
import gc as finalize_sideband_data_gc

import numpy as finalize_sideband_data_np
import pandas as finalize_sideband_data_pd

finalize_sideband_data_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_sideband_data_PROTECTED = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].PROTECTED; finalize_sideband_data_candidate_coordinates = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].candidate_coordinates
finalize_sideband_data_cached_atomic_measurements = _FINALIZE_NAMESPACES['abcd_sideband_validation'].cached_atomic_measurements; finalize_sideband_data_candidate_masks = _FINALIZE_NAMESPACES['abcd_sideband_validation'].candidate_masks; finalize_sideband_data_disclosure_audit = _FINALIZE_NAMESPACES['abcd_sideband_validation'].disclosure_audit


finalize_sideband_data_SIDEBANDS = tuple('BCD')
finalize_sideband_data_IDENTITY = ('candidate_id', 'strategy', 'a_iso', 'a_phi', 'b_iso', 'c_phi', 'display_label')
finalize_sideband_data_PREFERRED_ORDER = ('G_i0p3_p2', 'L_i0p3_p2p4', 'G_i0p3_p1p5', 'G_i2_p2')
finalize_sideband_data_SUMMARY_NUMERIC = (['data_' + r for r in finalize_sideband_data_SIDEBANDS] + ['var_' + r for r in finalize_sideband_data_SIDEBANDS]
                   + ['pred', 'pred_error', 'pred_rel_error'])


def finalize_sideband_data_run_nominal_sideband_data(config, tradeoff_result, evaluate_data=False):
    """Gate the MC shortlist jointly before optional safe nominal DATA queries.

    The order is fixed using MC before seeing any DATA. A candidate is released
    only when all three sidebands pass the 120-model purity gate, every sideband
    avoids the protected A union, and new identifiable complementary regions
    remain safe when combined with previously accepted candidates.
    """
    if (config['topology'], config['purity_limit'], config['partner_iso_low'],
            config['partner_iso_high']) != ('2mu2e', .1, 0., .5):
        raise ValueError('Require the established 2mu2e partner-isolation and 10% conventions.')
    selection = tradeoff_result['tables']['sideband_selection'].copy()
    if selection.candidate_id.duplicated().any():
        raise AssertionError('The MC-only shortlist must contain unique candidate IDs.')
    manifest, validation = [], []
    folder = finalize_sideband_data_Path(config['vr2_output_dir'])
    frames = []
    for filename in ('sr_reference_grid.csv', 'sr_reference_signal_points.csv'):
        path = folder / filename
        frames.append(finalize_sideband_data_pd.read_csv(path, float_precision='round_trip'))
        manifest.append(finalize_sideband_data_common.candidate_input_record(path, 'Nominal BCD disclosure audit: MC only'))
    background, signals = frames
    if len(background) != 154 or not background.source.eq('MC').all() or not background.context.eq('SR').all():
        raise AssertionError('The disclosure audit must use the 154-point nominal MC cache.')
    if len(signals) != 154*120 or signals['sample'].nunique() != 120 or not signals.context.eq('SR').all():
        raise AssertionError('The disclosure audit requires all 120 separate nominal signal hypotheses.')
    protected = list(finalize_sideband_data_PROTECTED)
    # Protect the A of operational loose candidates actually proposed for DATA,
    # not the A of every counterfactual MC scan point.
    protected.extend((float(row.a_iso), float(row.a_phi)) for row in selection.itertuples()
                     if row.strategy == 'loose' and bool(row.operational_pass))
    protected = sorted(set(protected))
    x_edges = sorted(set([0., finalize_sideband_data_np.inf] + selection.a_iso.tolist() + selection.b_iso.tolist()
                         + [point[0] for point in protected]))
    y_edges = sorted(set([-finalize_sideband_data_np.inf, finalize_sideband_data_np.inf] + selection.a_phi.tolist() + selection.c_phi.tolist()
                         + [point[1] for point in protected]))
    atomic = finalize_sideband_data_cached_atomic_measurements(background, signals, x_edges, y_edges)
    protected_mask = finalize_sideband_data_np.zeros(len(atomic['atoms']), bool)
    for x, y in protected:
        protected_mask |= finalize_sideband_data_candidate_masks(dict(a_iso=x, b_iso=x, a_phi=y, c_phi=y), atomic['atoms'])['A']
    lookup = tradeoff_result['tables']['sideband_regions'].set_index(['candidate_id', 'region'])
    proposed_order = {name: index for index, name in enumerate(finalize_sideband_data_PREFERRED_ORDER)}
    ordered = selection.assign(_order=selection.candidate_id.map(proposed_order).fillna(99))
    ordered = ordered.sort_values(['_order', 'selection_order'], kind='stable')
    published, approved, summary, regions, audit_rows = {}, {}, [], [], []
    for spec in ordered.to_dict('records'):
        identity = {name: spec[name] for name in finalize_sideband_data_IDENTITY}
        masks = finalize_sideband_data_candidate_masks(spec, atomic['atoms'])
        overlap = any((masks[region] & protected_mask).any() for region in finalize_sideband_data_SIDEBANDS)
        purity = []
        for region in finalize_sideband_data_SIDEBANDS:
            mask = masks[region]
            b, vb = float(atomic['values'][mask].sum()), float(atomic['variances'][mask].sum())
            signal = atomic['signals'][:, mask].sum(axis=1)
            ratios = signal/b if b > 1e-8 else finalize_sideband_data_np.full(120, finalize_sideband_data_np.nan)
            gate = bool(b > 1e-8 and finalize_sideband_data_np.isfinite(ratios).all() and (ratios < .1).all())
            purity.append(gate)
            cached = lookup.loc[(spec['candidate_id'], region)]
            finalize_sideband_data_np.testing.assert_allclose([b, vb], [cached.background, cached.var_background], rtol=1e-8, atol=1e-8)
            if b > 1e-8:
                finalize_sideband_data_np.testing.assert_allclose(finalize_sideband_data_np.max(ratios), cached.max_S_B, rtol=1e-8, atol=1e-10)
        preliminary = bool(all(purity) and not overlap and spec['operational_pass'])
        proposed = {spec['candidate_id'] + ':' + region: masks[region] for region in finalize_sideband_data_SIDEBANDS}
        audit = finalize_sideband_data_disclosure_audit(published, proposed, atomic) if preliminary else finalize_sideband_data_pd.DataFrame()
        complementary_pass = bool(preliminary and (audit.approved.all() if len(audit) else True))
        eligible = bool(preliminary and complementary_pass)
        reason = ('PURITY_GEOMETRY_AND_DISCLOSURE_PASS' if eligible else
                  'BCD_CONTAMINATION_AT_OR_ABOVE_10_PERCENT' if not all(purity) else
                  'PROTECTED_A_OVERLAP_MC_COUNTERFACTUAL_ONLY' if overlap else
                  'UNSAFE_OR_UNSUPPORTED_RECONSTRUCTABLE_COMPLEMENT')
        if len(audit):
            audit_rows.extend(dict(identity, **record) for record in audit.to_dict('records'))
        if eligible:
            published.update(proposed)
            approved[spec['candidate_id']] = identity
        summary.append(dict(identity, selection_order=spec['selection_order'],
                            purity_pass=all(purity), geometry_safe=not overlap,
                            complementary_pass=complementary_pass, release_eligible=eligible,
                            display_allowed=bool(eligible and evaluate_data),
                            reason=reason if not eligible or evaluate_data else 'DATA_EVALUATION_DISABLED',
                            release_reason=reason, max_S_B_BCD=spec['max_S_B_BCD'],
                            new_complement_checks=len(audit),
                            failed_complement_checks=int((~audit.approved).sum()) if len(audit) else 0,
                            nominal_A_evaluated=False, status='NOT_EVALUATED',
                            **{name: finalize_sideband_data_np.nan for name in finalize_sideband_data_SUMMARY_NUMERIC}))
        for region in finalize_sideband_data_SIDEBANDS:
            regions.append(dict(identity, region=region, value=finalize_sideband_data_np.nan, variance=finalize_sideband_data_np.nan, error=finalize_sideband_data_np.nan,
                                count=finalize_sideband_data_np.nan, display_allowed=bool(eligible and evaluate_data),
                                reason=reason if not eligible or evaluate_data else 'DATA_EVALUATION_DISABLED',
                                nominal_A_evaluated=False))
        validation.append(dict(scope=spec['candidate_id'],
                               check='Independent cumulative MC sidebands reproduce nominal cache yields, sumw2 and maximum contamination',
                               passed=True))
    data, data_manifest = {}, []
    if evaluate_data and approved:
        data, data_manifest = finalize_sideband_data__load_approved_sidebands(config, approved)
        manifest.extend(data_manifest)
    for row in summary:
        if row['display_allowed']:
            measurements = data[row['candidate_id']]
            for region in finalize_sideband_data_SIDEBANDS:
                row['data_' + region] = measurements[region]['value']
                row['var_' + region] = measurements[region]['variance']
            row.update(finalize_sideband_data_data_prediction(measurements))
    for row in regions:
        if row['display_allowed']:
            row.update(data[row['candidate_id']][row['region']])
    summary = finalize_sideband_data_pd.DataFrame(summary).sort_values('selection_order').reset_index(drop=True)
    regions = finalize_sideband_data_pd.DataFrame(regions)
    if not summary.loc[~summary.display_allowed, finalize_sideband_data_SUMMARY_NUMERIC].isna().all().all():
        raise AssertionError('Blocked candidate summaries must contain no DATA numbers.')
    if not regions.loc[~regions.display_allowed, ['value', 'variance', 'error', 'count']].isna().all().all():
        raise AssertionError('Blocked regional DATA must remain NaN.')
    validation.extend([
        dict(scope='Disclosure', check='Every new identifiable complementary region is gated before any original DATA file is read', passed=True),
        dict(scope='Protection', check='Every accepted B/C/D is disjoint from historical A union and operational selected loose A', passed=True),
        dict(scope='Blinding', check='No nominal A DATA index, full nominal DATA projection, kappa or DATA significance is evaluated', passed=True),
        dict(scope='Blinding', check='All blocked DATA fields remain NaN', passed=True),
    ])
    if data:
        validation.append(dict(scope='DATA', check='Every queried B/C/D bin is nonnegative integer DATA with sumw2 equal to count', passed=True))
        validation.append(dict(scope='Provenance', check='All 68 original DATA hashes and file sizes match the existing VR2 input manifest before querying', passed=True))
    return dict(tables=dict(sbd_summary=summary, sbd_regions=regions,
                           sbd_release_audit=finalize_sideband_data_pd.DataFrame(audit_rows),
                           sbd_validation=finalize_sideband_data_pd.DataFrame(validation), sbd_input_manifest=finalize_sideband_data_pd.DataFrame(manifest)),
                metadata=dict(nominal_A_evaluated=False, nominal_BCD_evaluated=bool(data),
                              original_data_files_opened=len(data_manifest),
                              source_hashes_match_existing_vr_manifest=bool(data),
                              evaluate_data=bool(evaluate_data), signal_hypotheses=120,
                              protected_A_boundaries=protected, approved_candidates=list(approved),
                              release_order=ordered.candidate_id.tolist(), previous_nominal_DATA_masks=0,
                              selection='DD; 0 <= partner iso < 0.5; MJJ >= 150 GeV including mass overflow',
                              protection='Historical reference/candidate A union plus operational selected loose A; counterfactual scanned A is not a DATA selection.',
                              query_policy='Approved B/C/D six-dimensional slices only; no A slice or complete nominal projection.',
                              prediction_policy='Uncorrected B*C/D with independent Poisson delta propagation; no closure correction or DATA-constrained significance.',
                              uncertainty='D=0 gives undefined prediction; B=0 or C=0 gives zero point estimate and undefined symmetric error; sparse intervals require a likelihood.'))


def finalize_sideband_data_data_prediction(measurements):
    """Return uncorrected ABCD prediction without any A measurement."""
    b, c, d = (measurements[region]['value'] for region in finalize_sideband_data_SIDEBANDS)
    vb, vc, vd = (measurements[region]['variance'] for region in finalize_sideband_data_SIDEBANDS)
    if d <= 0:
        return dict(pred=finalize_sideband_data_np.nan, pred_error=finalize_sideband_data_np.nan, pred_rel_error=finalize_sideband_data_np.nan, status='ZERO_D_UNDEFINED_PREDICTION')
    pred = b*c/d
    if b == 0 or c == 0:
        return dict(pred=pred, pred_error=finalize_sideband_data_np.nan, pred_rel_error=finalize_sideband_data_np.nan,
                    status='ZERO_B_OR_C_POINT_ESTIMATE_ONLY')
    variance = (c/d)**2*vb + (b/d)**2*vc + (b*c/d**2)**2*vd
    error = finalize_sideband_data_np.sqrt(variance)
    return dict(pred=pred, pred_error=error, pred_rel_error=error/pred,
                status='SPARSE_POISSON_DELTA_DIAGNOSTIC' if min(b, c, d) < 10 else 'POISSON_DELTA_DIAGNOSTIC')


def finalize_sideband_data_sideband_indices(edges, spec, region):
    """Create a native 6D B/C/D slice; an A request is an explicit error."""
    if region not in finalize_sideband_data_SIDEBANDS:
        raise ValueError('Only nominal B, C and D indices may be constructed.')
    finalize_sideband_data_candidate_coordinates(spec)
    split = finalize_sideband_data_common._comparison_split
    partner = slice(split(edges['iso1'], 0.), split(edges['iso1'], .5))
    disp0 = slice(None, split(edges['disp0'], 2.5))
    disp1 = slice(split(edges['disp1'], .5), None)
    mass = slice(split(edges['ljlj_mass'], 150.), None)
    iso = (slice(split(edges['iso0'], 0.), split(edges['iso0'], spec['a_iso'])) if region == 'C'
           else slice(split(edges['iso0'], spec['b_iso']), None))
    angle = (slice(split(edges['abs_dphi'], spec['a_phi']), None) if region == 'B'
             else slice(None, split(edges['abs_dphi'], spec['c_phi'])))
    return iso, partner, disp0, disp1, angle, mass


def finalize_sideband_data__load_approved_sidebands(config, approved):
    """Sum only prospective-audit-approved slices of each source DATA histogram."""
    background_dir = finalize_sideband_data_Path(config['bkg_dir'])
    data_dir = finalize_sideband_data_Path(config.get('data_dir', background_dir.with_name(background_dir.name.replace('_bkg_', '_data_'))))
    paths = sorted(data_dir.glob('*.coffea'))
    if len(paths) != 68:
        raise ValueError('Require the established 68 DATA files.')
    source_manifest = finalize_sideband_data_pd.read_csv(finalize_sideband_data_Path(config['vr2_output_dir']) / 'input_manifest.csv')
    source_manifest = source_manifest.loc[source_manifest.source.eq('DATA selected VR candidates')].set_index('path')
    if len(source_manifest) != 68 or set(map(str, paths)) != set(source_manifest.index):
        raise AssertionError('The original DATA file identities must match the existing VR2 production.')
    output = {name: {region: dict(value=0., variance=0.) for region in finalize_sideband_data_SIDEBANDS} for name in approved}
    manifest = []
    for index, path in enumerate(paths, 1):
        record = finalize_sideband_data_common.candidate_input_record(path, 'Prospectively approved nominal B/C/D DATA slices only; no A projection')
        previous = source_manifest.loc[str(path)]
        if record['sha256'] != previous.sha256 or record['size_bytes'] != previous.size_bytes:
            raise AssertionError('An original DATA file differs from the established VR2 input manifest.')
        loaded, payload = finalize_sideband_data_common._extension_data_payload(path)
        # The shared checker validates histogram schema, channel, flow, native
        # edges and storage; these are 6D array views, never a nominal projection.
        weights, variances, edges = finalize_sideband_data_common._comparison_current_cube(payload, path, config)
        for name, spec in approved.items():
            for region in finalize_sideband_data_SIDEBANDS:
                selection = finalize_sideband_data_sideband_indices(edges, spec, region)
                selected, selected_variances = weights[selection], variances[selection]
                if (not finalize_sideband_data_np.isfinite(selected).all() or (selected < 0).any()
                        or not finalize_sideband_data_np.array_equal(selected, selected_variances)
                        or not finalize_sideband_data_np.array_equal(selected, finalize_sideband_data_np.rint(selected))):
                    raise AssertionError('Approved DATA slices must have integer counts and identical sumw2.')
                output[name][region]['value'] += float(selected.sum())
                output[name][region]['variance'] += float(selected_variances.sum())
        manifest.append(record)
        del loaded, payload, weights, variances, selected, selected_variances
        finalize_sideband_data_gc.collect()
        if index % 20 == 0 or index == len(paths):
            print(f'09 approved nominal B/C/D DATA {index}/{len(paths)}', flush=True)
    for measurement in output.values():
        for row in measurement.values():
            row.update(error=float(finalize_sideband_data_np.sqrt(row['variance'])), count=int(round(row['value'])))
    return output, manifest


# ========================================================================
# abcd_final_nominal_controls: original study API
# ========================================================================
"""Evaluate the final nominal B/C/D definitions while keeping selected A blind.

The historical union of discarded candidate A regions is deliberately not a
constraint. The MC-selected final loose A remains protected, including the
nested WP2 A used by the buffer comparison. Only B/C/D slices are ever summed.
"""
import numpy as finalize_final_nominal_controls_np
import pandas as finalize_final_nominal_controls_pd

finalize_final_nominal_controls_IDENTITY = _FINALIZE_NAMESPACES['abcd_sideband_data'].IDENTITY; finalize_final_nominal_controls__load_approved_sidebands = _FINALIZE_NAMESPACES['abcd_sideband_data']._load_approved_sidebands; finalize_final_nominal_controls_data_prediction = _FINALIZE_NAMESPACES['abcd_sideband_data'].data_prediction
finalize_final_nominal_controls__overlap = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff']._overlap; finalize_final_nominal_controls_region_rectangles = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].region_rectangles


def finalize_final_nominal_controls_evaluate_final_nominal_controls(config, iso_result, previous_tables,
                                    extra_specs=None, evaluate_data=False):
    """Read the selected loose controls and optional final buffer controls.

    The explicit evaluation switch is set by notebook 09 following the user's
    final-definition instruction. The shortlist is fixed with MC before DATA
    are opened; observed counts and signal-fraction diagnostics do not rerank it.
    """
    if (config['topology'], config['purity_limit'], config['partner_iso_low'],
            config['partner_iso_high']) != ('2mu2e', .1, 0., .5):
        raise ValueError('Require 2mu2e, strict 10% MC purity and partner isolation [0,.5).')
    selected = iso_result['tables']['if_selection']
    selected = selected.loc[selected.role.eq('global_best')]
    if len(selected) != 1:
        raise AssertionError('One MC-selected final loose definition is required.')
    selected = selected.iloc[0]
    if selected.candidate_id != 'I_i0p25_p2p6':
        raise AssertionError('The reviewed final loose candidate must be (0.25,2.6).')
    loose = {key: selected[key] for key in finalize_final_nominal_controls_IDENTITY}
    loose.update(candidate_id=selected.source_candidate_id, strategy='loose',
                 display_label='Final loose (0.25,2.6)')
    specs = [loose] + list(extra_specs or [])
    if len({spec['candidate_id'] for spec in specs}) != len(specs):
        raise AssertionError('Final nominal candidates must be unique.')
    final_A = finalize_final_nominal_controls_region_rectangles(loose)['A']
    summary = previous_tables['sbd_summary'].copy()
    regions = previous_tables['sbd_regions'].copy()
    for frame in (summary, regions):
        frame['new_data_evaluated'] = False
    old_public = regions.loc[regions.display_allowed]
    if old_public.nominal_A_evaluated.any() or not old_public.region.isin(list('BCD')).all():
        raise AssertionError('Previous nominal observations must contain B/C/D only.')
    for row in old_public.to_dict('records'):
        if finalize_final_nominal_controls__overlap(finalize_final_nominal_controls_region_rectangles(row)[row['region']], final_A):
            raise AssertionError('An existing nominal observation overlaps the final A.')
    mc = previous_tables['sideband_regions']
    signals = previous_tables['sideband_signals']
    validation, checks, pending, reused = [], [], {}, {}
    for spec in specs:
        candidate = spec['candidate_id']
        rectangle = finalize_final_nominal_controls_region_rectangles(spec)
        if any(finalize_final_nominal_controls__overlap(rectangle[region], final_A) for region in 'BCD'):
            raise AssertionError('Final control request intersects selected nominal A: '+candidate)
        if not (rectangle['A'][0] >= final_A[0] and rectangle['A'][1] <= final_A[1]
                and rectangle['A'][2] >= final_A[2] and rectangle['A'][3] <= final_A[3]):
            raise AssertionError('Each compared SR must be inside the blinded final loose A.')
        source = mc.loc[mc.candidate_id.eq(candidate)].set_index('region')
        models = signals.loc[signals.candidate_id.eq(candidate)].sort_values('sample')
        if len(source) != 4 or len(models) != 120 or not models['sample'].is_unique:
            raise AssertionError('Require exact regional MC and 120 separate signal hypotheses.')
        for field in ('a_iso', 'a_phi', 'b_iso', 'c_phi'):
            if not finalize_final_nominal_controls_np.allclose(source[field], spec[field], rtol=0, atol=1e-12):
                raise AssertionError('MC control geometry differs from final request.')
        maxima = []
        for region in 'BCD':
            b = float(source.loc[region, 'background'])
            vb = float(source.loc[region, 'var_background'])
            s = models['S_'+region].to_numpy(float)
            if b <= 1e-8 or vb < 0 or not finalize_final_nominal_controls_np.isfinite([b, vb]).all() or not finalize_final_nominal_controls_np.isfinite(s).all() or (s < 0).any():
                raise AssertionError('Unsupported MC in final control: '+candidate+':'+region)
            maximum = float(finalize_final_nominal_controls_np.max(s/b))
            if maximum >= .1:
                raise AssertionError('Final control fails all-120 central MC contamination: '+candidate+':'+region)
            finalize_final_nominal_controls_np.testing.assert_allclose(maximum, source.loc[region, 'max_S_B'], rtol=1e-9, atol=1e-10)
            maxima.append(maximum)
            checks.append(dict(candidate_id=candidate, region=region,
                max_S_B_MC=maximum, purity_pass=True, overlap_final_A=False,
                historical_A_veto_applied=False, nominal_A_evaluated=False))
        existing = old_public.loc[old_public.candidate_id.eq(candidate)]
        if len(existing) == 3:
            reused[candidate] = {row['region']: dict(value=row['value'], variance=row['variance'],
                error=row['error'], count=int(row['count'])) for row in existing.to_dict('records')}
        elif len(existing):
            raise AssertionError('Partial previous nominal controls are not an exact reusable definition.')
        else:
            pending[candidate] = dict(spec)
        spec['max_S_B_BCD'] = max(maxima)
    validation.extend([
        dict(scope='Final nominal selection', check='Selected loose point is fixed by absolute MC significance and all-120 purity before DATA; no historical-A veto', passed=True),
        dict(scope='Final nominal blindness', check='All prior public and newly proposed nominal controls are disjoint from final loose A; compared buffer A is nested inside it', passed=True),
        dict(scope='Final nominal purity', check='Every requested B/C/D has positive MC support and S/B_MC<10% for all 120 hypotheses', passed=True),
    ])
    new_data, inputs = finalize_final_nominal_controls__load_approved_sidebands(config, pending) if evaluate_data and pending else ({}, [])
    measurements = {**reused, **new_data}
    new_summary, new_regions = [], []
    for order, spec in enumerate(specs, 1):
        candidate = spec['candidate_id']
        identity = {key: spec[key] for key in finalize_final_nominal_controls_IDENTITY}
        available, fresh = candidate in measurements, candidate in new_data
        reason = ('FINAL_SELECTED_BCD_EVALUATED_WITH_A_BLINDED' if fresh else
                  'EXACT_PREVIOUS_NOMINAL_BCD_REUSED' if available else 'FINAL_DATA_EVALUATION_DISABLED')
        row = dict(identity, selection_order=order, purity_pass=True, geometry_safe=True,
            complementary_pass=finalize_final_nominal_controls_np.nan, historical_complement_veto_applied=False,
            release_eligible=True, display_allowed=available, reason=reason, release_reason=reason,
            max_S_B_BCD=spec['max_S_B_BCD'], new_complement_checks=0, failed_complement_checks=0,
            nominal_A_evaluated=False, new_data_evaluated=fresh, status='NOT_EVALUATED',
            pred=finalize_final_nominal_controls_np.nan, pred_error=finalize_final_nominal_controls_np.nan, pred_rel_error=finalize_final_nominal_controls_np.nan)
        for region in 'BCD':
            values = measurements[candidate][region] if available else dict(value=finalize_final_nominal_controls_np.nan,
                variance=finalize_final_nominal_controls_np.nan, error=finalize_final_nominal_controls_np.nan, count=finalize_final_nominal_controls_np.nan)
            row['data_'+region], row['var_'+region] = values['value'], values['variance']
            new_regions.append(dict(identity, region=region, **values, display_allowed=available,
                reason=reason, nominal_A_evaluated=False, new_data_evaluated=fresh))
        if available:
            row.update(finalize_final_nominal_controls_data_prediction(measurements[candidate]))
        new_summary.append(row)
    ids = [spec['candidate_id'] for spec in specs]
    summary = finalize_final_nominal_controls_pd.concat([summary.loc[~summary.candidate_id.isin(ids)], finalize_final_nominal_controls_pd.DataFrame(new_summary)], ignore_index=True)
    regions = finalize_final_nominal_controls_pd.concat([regions.loc[~regions.candidate_id.isin(ids)], finalize_final_nominal_controls_pd.DataFrame(new_regions)], ignore_index=True)
    validation.append(dict(scope='Final nominal DATA',
        check='Only B/C/D are constructed and summed; no nominal A yield, total or projection is evaluated', passed=True))
    return dict(tables=dict(sbd_summary=summary, sbd_regions=regions,
        nominal_final_gate=finalize_final_nominal_controls_pd.DataFrame(checks), nominal_final_validation=finalize_final_nominal_controls_pd.DataFrame(validation),
        nominal_final_input_manifest=finalize_final_nominal_controls_pd.DataFrame(inputs)), metadata=dict(
        selected_loose='L_i0p25_p2p6', requested_candidates=ids,
        evaluated_candidates=list(measurements), new_candidates=list(new_data),
        original_data_files_opened=len(inputs), nominal_A_evaluated=False,
        historical_A_union_used=False, protected_final_A=dict(a_iso=.25, a_phi=2.6),
        historical_complement_veto_applied=False,
        data_used_for_selection=False))


# ========================================================================
# abcd_sideband_scan_validation: original study API
# ========================================================================
"""Native loose/gap VR scans with joint mass and prior-disclosure protection.

The full scan is MC-only.  DATA are considered only on prespecified 1D paths,
after a joint all-mass/fine-partition audit including both earlier 09 reports.
Nominal DATA and DD above 150 GeV are never selected by this module.
"""
from pathlib import Path as finalize_sideband_scan_validation_Path
import gc as finalize_sideband_scan_validation_gc
import hashlib as finalize_sideband_scan_validation_hashlib
import json as finalize_sideband_scan_validation_json

import numpy as finalize_sideband_scan_validation_np
import pandas as finalize_sideband_scan_validation_pd
from threadpoolctl import threadpool_limits as finalize_sideband_scan_validation_threadpool_limits

finalize_sideband_scan_validation_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_sideband_scan_validation_previous = _FINALIZE_NAMESPACES['abcd_sideband_validation']
finalize_sideband_scan_validation_candidate_coordinates = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].candidate_coordinates; finalize_sideband_scan_validation_statistics_from_regions = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].statistics_from_regions
finalize_sideband_scan_validation_project_mass_partitions = _FINALIZE_NAMESPACES['abcd_mass_inclusive'].project_mass_partitions; finalize_sideband_scan_validation__state_index = _FINALIZE_NAMESPACES['abcd_mass_inclusive']._state_index
finalize_sideband_scan_validation_strata_specs = _FINALIZE_NAMESPACES['abcd_final_check_strata']._specs; finalize_sideband_scan_validation_project_strata = _FINALIZE_NAMESPACES['abcd_final_check_strata']._project_mc


finalize_sideband_scan_validation_REGIONS = tuple('ABCD')
finalize_sideband_scan_validation_IDENTITY = ('candidate_id', 'strategy', 'a_iso', 'a_phi', 'b_iso', 'c_phi', 'display_label')
finalize_sideband_scan_validation_SCOPES = [('VR1_DD', 'low', 'DD'), ('VR1_PP', 'low', 'PP'), ('VR1_PD', 'low', 'PD'),
          ('VR1_DP', 'low', 'DP'), ('VR2_PP', 'intermediate', 'PP'),
          ('VR2_PD', 'intermediate', 'PD'), ('VR2_DP', 'intermediate', 'DP'),
          ('INCLUSIVE_PP', 'all', 'PP'), ('INCLUSIVE_PD', 'all', 'PD'),
          ('INCLUSIVE_DP', 'all', 'DP')]


def finalize_sideband_scan_validation_run_scan_vr_validation(config, nominal_result, historical_tables, previous_tables,
                           load_data=False, mc=None):
    """Compute all MC candidates; release only audited 1D-path DATA observations."""
    if (config['topology'], config['purity_limit'], config['partner_iso_low'],
            config['partner_iso_high']) != ('2mu2e', .1, 0., .5):
        raise ValueError('Require established 2mu2e, partner [0,.5), and strict 10% conventions.')
    inputs, validation = [], []
    finalize_sideband_scan_validation__audit_prior_snapshots(config, historical_tables, previous_tables, inputs, validation)
    candidates = nominal_result['tables']['sideband_scan'].copy()
    if len(candidates) != 131 or candidates.candidate_id.duplicated().any():
        raise AssertionError('Require the 64+64 native scan and three original baseline definitions.')
    paths = finalize_sideband_scan_validation_scan_paths(candidates)
    requested = set(paths.candidate_id) | set(nominal_result['tables']['sideband_selection'].candidate_id)
    caches = finalize_sideband_scan_validation_previous._load_caches(config, inputs, validation)
    mc = finalize_sideband_scan_validation_load_scan_mc(config) if mc is None else mc
    inputs.extend(mc['manifest'])
    validation.extend(mc['validation'])
    finalize_sideband_scan_validation__validate_mc_inputs(mc, caches, config, validation, inputs)
    finalize_sideband_scan_validation__validate_prior_fine_mc(mc, historical_tables, validation)
    finalize_sideband_scan_validation__validate_joint_complement_safety(validation)
    x_edges = sorted(set([0., finalize_sideband_scan_validation_np.inf] + candidates.a_iso.tolist() + candidates.b_iso.tolist()))
    y_edges = sorted(set([-finalize_sideband_scan_validation_np.inf, finalize_sideband_scan_validation_np.inf] + candidates.a_phi.tolist() + candidates.c_phi.tolist()))
    # The release order is fixed from nominal MC roles before any new DATA value.
    priorities = ['L_i0p2_p1p5', 'L_i0p3_p2p4', 'G_i0p3_p2', 'WP2', 'WP1', 'Reference',
                  'G_i0p3_p1p5', 'G_i2_p2', 'L_i0p25_p2p6']
    ranks = {name: i for i, name in enumerate(priorities)}
    ordered = candidates.assign(_order=candidates.candidate_id.map(ranks).fillna(99)).sort_values(
        ['_order', 'native_steps', 'candidate_id'], kind='stable')
    closure, purity, releases, complements, approved_inclusive = [], [], [], [], []
    exhaustive_rows = []
    published_by_state, atomic_by_state = {}, {}
    for state in ('DD', 'PP', 'PD', 'DP'):
        atomic = finalize_sideband_scan_validation__joint_atomic(mc, caches, state, x_edges, y_edges, validation)
        published = finalize_sideband_scan_validation__prior_masks(historical_tables, previous_tables, state, atomic)
        guard = finalize_sideband_scan_validation__exhaustive_guard(atomic, published, x_edges, y_edges)
        published_by_state[state] = published
        atomic_by_state[state] = atomic
        initial_count = len(published)
        scopes = [entry for entry in finalize_sideband_scan_validation_SCOPES if entry[2] == state]
        # Low and intermediate scans are evaluated before inclusive to prevent
        # the later release of an unsafe high-mass remainder by subtraction.
        for scope, mass, _ in scopes:
            print(f'09 joint VR scan audit: {scope}', flush=True)
            plane = atomic['atoms']
            for spec in ordered.to_dict('records'):
                identity = {key: spec[key] for key in finalize_sideband_scan_validation_IDENTITY}
                identity.update(scope=scope, plane=state, mass_slice=mass)
                masks = finalize_sideband_scan_validation__scope_masks(spec, atomic, mass)
                values = {r: float(atomic['values'][masks[r]].sum()) for r in finalize_sideband_scan_validation_REGIONS}
                variances = {r: float(atomic['variances'][masks[r]].sum()) for r in finalize_sideband_scan_validation_REGIONS}
                stat = finalize_sideband_scan_validation_statistics_from_regions(values, variances, config.get('min_neff', 10.))
                gate_rows = []
                for region in finalize_sideband_scan_validation_REGIONS:
                    signals = atomic['signals'][:, masks[region]].sum(axis=1)
                    ratios = finalize_sideband_scan_validation_previous._ratios(signals, values[region])
                    valid = bool(values[region] > 1e-8 and finalize_sideband_scan_validation_np.isfinite(ratios).all()
                                 and finalize_sideband_scan_validation_np.all(signals >= -1e-10))
                    worst = int(finalize_sideband_scan_validation_np.argmax(signals))
                    maximum = float(ratios[worst]) if valid else finalize_sideband_scan_validation_np.nan
                    gate_rows.append(dict(identity, region=region, background=values[region],
                        var_background=variances[region], max_S_B=maximum,
                        worst_sample=atomic['signal_names'][worst], signal_hypotheses=120,
                        approved=bool(valid and maximum < .1), valid_background=valid))
                purity.extend(gate_rows)
                own_pass = all(row['approved'] for row in gate_rows)
                maximum = max(row['max_S_B'] for row in gate_rows) if own_pass or all(
                    finalize_sideband_scan_validation_np.isfinite(row['max_S_B']) for row in gate_rows) else finalize_sideband_scan_validation_np.nan
                on_path = spec['candidate_id'] in requested
                source_pass, source_rows = finalize_sideband_scan_validation__cached_sources(caches, mass, state, spec)
                proposed = {scope + ':' + spec['candidate_id'] + ':' + r: masks[r] for r in finalize_sideband_scan_validation_REGIONS}
                audit = finalize_sideband_scan_validation_joint_disclosure_audit(published, proposed, atomic) if on_path and own_pass and source_pass else finalize_sideband_scan_validation_pd.DataFrame()
                guard_probe = None
                if on_path and own_pass and source_pass:
                    guard_probe = finalize_sideband_scan_validation__guard_probe(guard, proposed)
                    guard_audit = guard_probe['audit']
                    if len(guard_audit):
                        audit = finalize_sideband_scan_validation_pd.concat([audit, guard_audit], ignore_index=True)
                complement_pass = bool(audit.approved.all()) if len(audit) else bool(on_path and own_pass)
                eligible = bool(on_path and own_pass and source_pass and complement_pass)
                reason = ('PURITY_AND_JOINT_MASS_DISCLOSURE_PASS' if eligible else
                          'MC_SCAN_ONLY_NOT_ON_PRESPECIFIED_DATA_PATH' if not on_path else
                          'VR_CONTAMINATION_OR_INVALID_BACKGROUND' if not own_pass else
                          'SOURCE_CACHE_BLOCKED' if not source_pass else
                          'UNSAFE_OR_UNSUPPORTED_JOINT_MASS_COMPLEMENT')
                if len(audit):
                    complements.extend(dict(identity, **record) for record in audit.to_dict('records'))
                release = dict(identity, approved=own_pass, max_S_B=maximum, signal_hypotheses=120,
                    on_data_path=on_path, source_cache_approved=source_pass,
                    complementary_pass=complement_pass, release_eligible=eligible,
                    display_allowed=bool(eligible and load_data), reason=reason,
                    new_complement_checks=len(audit),
                    failed_complement_checks=int((~audit.approved).sum()) if len(audit) else 0,
                    previous_public_masks=initial_count)
                releases.append(release)
                closure.append(dict(identity, **stat, source='MC', approved=own_pass,
                    max_S_B=maximum, signal_hypotheses=120, display_allowed=True,
                    release_eligible=eligible, reason='MC_ALWAYS_VISIBLE'))
                blank = {field: finalize_sideband_scan_validation_np.nan for field in finalize_sideband_scan_validation_previous.MEASUREMENT_FIELDS}
                blank.update(status='BLINDED', kappa_status='BLINDED', low_stat=False)
                if eligible and load_data and mass != 'all':
                    vals = {r: float(source_rows[r][r]) for r in finalize_sideband_scan_validation_REGIONS}
                    vari = {r: float(source_rows[r]['var_' + r]) for r in finalize_sideband_scan_validation_REGIONS}
                    if not finalize_sideband_scan_validation_np.isfinite(list(vals.values()) + list(vari.values())).all():
                        raise AssertionError('Approved source cache has missing DATA counts.')
                    finalize_sideband_scan_validation_np.testing.assert_array_equal(list(vals.values()), list(vari.values()))
                    blank = finalize_sideband_scan_validation_data_statistics(vals, vari, config.get('min_neff', 10.))
                record = dict(identity, **blank, source='DATA', approved=own_pass,
                    max_S_B=maximum, signal_hypotheses=120, display_allowed=bool(eligible and load_data),
                    release_eligible=eligible, reason=reason if load_data or not eligible else 'DATA_EVALUATION_DISABLED')
                closure.append(record)
                if eligible:
                    finalize_sideband_scan_validation__guard_commit(guard, guard_probe)
                    published.update(proposed)
                    published = finalize_sideband_scan_validation__unique_masks(published)
                    if mass == 'all':
                        approved_inclusive.append(identity)
                if mass != 'all':
                    finalize_sideband_scan_validation__regress_candidate_mc(caches, mass, state, spec, values, variances, validation)
        validation.append(dict(scope=state,
            check='Joint disclosure audit includes both prior reports and every newly eligible low/intermediate/inclusive observation', passed=True))
        independent_unsafe = finalize_sideband_scan_validation__validate_final_exhaustive_basis(guard, published)
        exhaustive_rows.append(dict(plane=state, rectangle_mass_subsets=guard['total_rectangles'],
            unsafe_MC_regions=guard['total_unsafe'], previously_known_unsafe=guard['grandfathered'],
            proposed_queries_checked=guard['probes'], rejected_query_sets=guard['rejected'],
            newly_identifiable_unsafe_after_release=int((guard['residual'] < 1e-7).sum()),
            full_QR_recheck_newly_identifiable_unsafe=independent_unsafe))
        if exhaustive_rows[-1]['newly_identifiable_unsafe_after_release']:
            raise AssertionError('The final accepted release resolves an unsafe native rectangle.')
        validation.append(dict(scope=state,
            check='Exhaustive native rectangles times every nonempty mass/prompt-leaf subset contain no newly identifiable unsafe region', passed=True))
        validation.append(dict(scope=state,
            check='Independent full public-mask QR decomposition confirms the incremental exhaustive disclosure result', passed=True))
    # No original DATA access occurs until every release decision has been made.
    inclusive_data, data_manifest = ({}, [])
    if load_data and approved_inclusive:
        inclusive_data, data_manifest = finalize_sideband_scan_validation__load_inclusive_data(config, approved_inclusive)
        inputs.extend(data_manifest)
        for row in closure:
            if row['source'] == 'DATA' and row['mass_slice'] == 'all' and row['display_allowed']:
                counts = inclusive_data[(row['plane'], row['candidate_id'])]
                row.update(finalize_sideband_scan_validation_data_statistics(counts, counts, config.get('min_neff', 10.)))
    tables = dict(sv_closure=finalize_sideband_scan_validation_pd.DataFrame(closure), sv_purity=finalize_sideband_scan_validation_pd.DataFrame(purity),
                  sv_release_audit=finalize_sideband_scan_validation_pd.DataFrame(releases), sv_complements=finalize_sideband_scan_validation_pd.DataFrame(complements),
                  sv_paths=paths, sv_exhaustive_audit=finalize_sideband_scan_validation_pd.DataFrame(exhaustive_rows),
                  sv_input_manifest=finalize_sideband_scan_validation_pd.DataFrame(inputs), sv_validation=finalize_sideband_scan_validation_pd.DataFrame(validation))
    hidden = tables['sv_closure'].query("source == 'DATA' and not display_allowed")
    if not hidden[finalize_sideband_scan_validation_previous.MEASUREMENT_FIELDS].isna().all().all():
        raise AssertionError('Every blocked or non-path DATA measurement must remain NaN.')
    if load_data:
        finalize_sideband_scan_validation__regress_previous_data(tables['sv_closure'], previous_tables, validation)
        shown = tables['sv_closure'].query("source == 'DATA' and display_allowed")
        empty_numerator = shown.loc[(shown.B.eq(0) | shown.C.eq(0)) & shown.D.gt(0)]
        if not empty_numerator.pred.eq(0).all() or not empty_numerator.pred_error.isna().all():
            raise AssertionError('An empty DATA numerator must not have a zero symmetric prediction error.')
        validation.append(dict(scope='Sparse DATA',
            check='Empty B/C numerator keeps a zero point prediction and undefined symmetric error; empty D remains undefined', passed=True))
    validation.extend([
        dict(scope='Blinding', check='No nominal DATA or DD with MJJ>=150 is selected; inclusive DATA use only approved explicit P-containing regions', passed=True),
        dict(scope='DATA', check='Every blocked yield, prediction, kappa and error remains NaN', passed=True),
        dict(scope='Paths', check='New DATA paths and release priorities are fixed using nominal MC before VR observations', passed=True),
    ])
    tables['sv_validation'] = finalize_sideband_scan_validation_pd.DataFrame(validation)
    return dict(tables=tables, metadata=dict(signal_hypotheses=120, nominal_A_evaluated=False,
        nominal_BCD_evaluated=False, original_data_files_opened=len(data_manifest),
        inclusive_data_evaluated=bool(inclusive_data), cached_vr_data_evaluated=bool(load_data),
        scopes=[row[0] for row in finalize_sideband_scan_validation_SCOPES], total_candidates=len(candidates),
        data_path_candidates=len(requested), release_priorities=priorities,
        exhaustive_rectangle_mass_subsets=sum(row['rectangle_mass_subsets'] for row in exhaustive_rows),
        exhaustive_newly_identifiable_unsafe=sum(row['newly_identifiable_unsafe_after_release'] for row in exhaustive_rows),
        full_QR_recheck_newly_identifiable_unsafe=sum(row['full_QR_recheck_newly_identifiable_unsafe'] for row in exhaustive_rows),
        inclusive_definition='MJJ>=0 including overflow; only explicit PP/PD/DP; no inclusive displacement category.',
        disclosure_definition='Joint low/intermediate/high MC atoms, native 150-200/200-300 and PD prompt leaves, all prior public observations and new scan releases.',
        high_mass_scope='MC is evaluated only to protect reconstructable high-mass remainders; high-mass DATA counts are never selected or reported.',
        selection_policy='Prespecified 1D paths; DATA do not determine candidate ordering or nominal sensitivity.',
        statistical_policy='Disjoint sumw2/delta propagation; overlapping scan points are correlated and sparse symmetric intervals are diagnostic.'))


def finalize_sideband_scan_validation_scan_paths(candidates):
    definitions = [
        ('loose_iso_phi1p5', 'loose', 'a_iso', 'a_phi', 1.5),
        ('loose_iso_phi2p4', 'loose', 'a_iso', 'a_phi', 2.4),
        ('loose_iso_phi2p7', 'loose', 'a_iso', 'a_phi', 2.7),
        ('loose_phi_iso0p2', 'loose', 'a_phi', 'a_iso', .2),
        ('loose_phi_iso0p3', 'loose', 'a_phi', 'a_iso', .3),
        ('gap_iso_phi2', 'gap', 'b_iso', 'c_phi', 2.),
        ('gap_phi_iso0p3', 'gap', 'c_phi', 'b_iso', .3)]
    rows = []
    for name, strategy, axis, fixed, value in definitions:
        part = candidates.loc[candidates.strategy.eq(strategy) & finalize_sideband_scan_validation_np.isclose(candidates[fixed], value)]
        if len(part) != 8:
            raise AssertionError('Each native scan path must contain eight candidates.')
        for row in part.sort_values(axis).to_dict('records'):
            rows.append(dict(candidate_id=row['candidate_id'], strategy=strategy, path_id=name,
                scan_axis=axis, scan_value=row[axis], fixed_axis=fixed, fixed_value=value))
    return finalize_sideband_scan_validation_pd.DataFrame(rows)


def finalize_sideband_scan_validation__audit_prior_snapshots(config, historical, last, inputs, validation):
    base = finalize_sideband_scan_validation_Path(config['notebook_path']).parent
    for folder in ('09_20261026_reference', '09_20261027_sideband_reference'):
        archive = base / folder
        manifest = archive / 'archive_manifest.json'
        document = finalize_sideband_scan_validation_json.loads(manifest.read_text())
        for relative, digest in document['files'].items():
            if finalize_sideband_scan_validation_hashlib.sha256((archive / relative).read_bytes()).hexdigest() != digest:
                raise AssertionError('Historical release archive changed: ' + relative)
        inputs.append(finalize_sideband_scan_validation_common.candidate_input_record(manifest, 'Joint VR release prior archive'))
    for table, frames, folder in [
        ('mass_closure', historical, '09_20261026_reference'),
        ('strata_closure', historical, '09_20261026_reference'),
        ('report_lowmass_dd', historical, '09_20261026_reference'),
        ('sbvr_closure', last, '09_20261027_sideband_reference'),
        ('sbd_regions', last, '09_20261027_sideband_reference')]:
        path = base / folder / '09_2mu2e_final_check_outputs' / (table + '.csv')
        expected = finalize_sideband_scan_validation_pd.read_csv(path, float_precision='round_trip')
        finalize_sideband_scan_validation_pd.testing.assert_frame_equal(frames[table].reset_index(drop=True), expected, check_dtype=False)
        inputs.append(finalize_sideband_scan_validation_common.candidate_input_record(path, 'Previously displayed DATA masks'))
    validation.append(dict(scope='Archives', check='Both prior 09 snapshots match their SHA256 manifests and supplied release tables', passed=True))


def finalize_sideband_scan_validation_load_scan_mc(config):
    """Read the 18+120 original MC files once, preserving fine and high-mass atoms."""
    bkg = finalize_sideband_scan_validation_Path(config['bkg_dir'])
    settings = dict(config, bkg_dir=bkg, signal_dir=bkg.with_name(bkg.name.replace('_bkg_', '_signal_')))
    backgrounds = sorted(bkg.glob('*.coffea'))
    signals = finalize_sideband_scan_validation_common.candidate_signal_paths(settings)
    if len(backgrounds) != 18 or len(signals) != 120:
        raise AssertionError('Require the established 18 background and 120 signal MC samples.')
    specs = finalize_sideband_scan_validation_strata_specs()
    totals, signal_planes, manifest, checks = {}, {}, [], []
    jobs = [(path, 'Background') for path in backgrounds] + [(path, 'Signal') for path in signals]
    for index, (path, source) in enumerate(jobs, 1):
        loaded, payload = finalize_sideband_scan_validation_common._comparison_payload(path, 'sixd')
        weights, variances, edges = finalize_sideband_scan_validation_common._comparison_current_cube(payload, path, settings)
        fine = finalize_sideband_scan_validation_project_strata(weights, variances, edges, specs, settings)
        planes = {'fine:' + name: plane for name, plane in fine.items()}
        for state in ('PP', 'PD', 'DP'):
            partitions, _ = finalize_sideband_scan_validation_project_mass_partitions(weights, variances, edges, state, settings)
            planes.update({state + ':' + mass: plane for mass, plane in partitions.items()})
        for name, plane in planes.items():
            if source == 'Background':
                totals[name] = finalize_sideband_scan_validation_common.add_candidate(totals.get(name), plane, finalize_sideband_scan_validation_common._comparison_group(path.stem))
            else:
                signal_planes.setdefault(name, {})[path.stem] = plane
        manifest.append(finalize_sideband_scan_validation_common.candidate_input_record(path, 'VR scan native ' + source + ' MC'))
        checks.append(dict(scope=path.stem, check='Original MC fine-mass/prompt and low/intermediate/high projections conserve native sumw/sumw2', passed=True))
        del loaded, payload, weights, variances, fine, planes
        finalize_sideband_scan_validation_gc.collect()
        if index % 20 == 0 or index == len(jobs):
            print(f'09 full VR scan MC {index}/{len(jobs)}', flush=True)
    return dict(backgrounds=totals, signals=signal_planes, manifest=manifest, validation=checks)


def finalize_sideband_scan_validation__validate_mc_inputs(mc, caches, config, validation, inputs):
    for mass in ('low', 'intermediate'):
        cache = caches[mass]
        for state in ('PP', 'PD', 'DP'):
            plane = mc['backgrounds'][state + ':' + mass]
            rows = cache['grid'].query("source == 'MC' and plane == @state")
            for row in rows.itertuples():
                actual = finalize_sideband_scan_validation_common._comparison_regions(plane, row.tx, row.ty)
                for r in finalize_sideband_scan_validation_REGIONS:
                    finalize_sideband_scan_validation_np.testing.assert_allclose([actual[r]['value'], actual[r]['variance']],
                        [getattr(row, r), getattr(row, 'var_' + r)], rtol=1e-8, atol=1e-8)
            signal_rows = cache['signals'].loc[cache['signals'].plane.eq(state)]
            for row in signal_rows.itertuples():
                actual = finalize_sideband_scan_validation_common._comparison_regions(mc['signals'][state + ':' + mass][row.sample], row.tx, row.ty)
                for r in finalize_sideband_scan_validation_REGIONS:
                    finalize_sideband_scan_validation_np.testing.assert_allclose([actual[r]['value'], actual[r]['variance']],
                        [getattr(row, 'S_' + r), getattr(row, 'var_S_' + r)], rtol=1e-8, atol=1e-8)
            validation.append(dict(scope=mass + '_' + state,
                check='Raw MC reproduces all 154 cached background points and all 120 signals per point', passed=True))
    path = finalize_sideband_scan_validation_Path(config['comprehensive_dir']) / 'mass_grid.csv'
    all_mass = finalize_sideband_scan_validation_pd.read_csv(path, float_precision='round_trip').query("source == 'MC'")
    inputs.append(finalize_sideband_scan_validation_common.candidate_input_record(path, 'Inclusive background MC regression only'))
    for row in all_mass.itertuples():
        actual = finalize_sideband_scan_validation_common._comparison_regions(mc['backgrounds'][row.plane + ':all'], row.tx, row.ty)
        for r in finalize_sideband_scan_validation_REGIONS:
            finalize_sideband_scan_validation_np.testing.assert_allclose([actual[r]['value'], actual[r]['variance']],
                [getattr(row, r), getattr(row, 'var_' + r)], rtol=1e-8, atol=1e-8)
    validation.append(dict(scope='Inclusive', check='All 462 inclusive background MC coordinates reproduce notebook 07', passed=True))


def finalize_sideband_scan_validation__joint_atomic(mc, caches, state, x_edges, y_edges, validation):
    background = caches['low']['grid'].query("source == 'MC' and plane == @state")
    signals = caches['low']['signals'].loc[caches['low']['signals'].plane.eq(state)]
    low = finalize_sideband_scan_validation_previous.cached_atomic_measurements(background, signals, x_edges, y_edges)
    atoms, names = low['atoms'], low['signal_names']
    leaves = [(0., 150., 'all', state + ':low')]
    if state == 'PP':
        leaves += [(150., 200., 'all', 'fine:PP_150_200_all'),
                   (200., 300., 'all', 'fine:PP_200_300_all')]
    elif state == 'PD':
        leaves += [(lo, hi, prompt, f'fine:PD_{lo}_{hi}_{prompt}')
                   for lo, hi in [(150, 200), (200, 300)] for prompt in ('3', 'ge4')]
    elif state == 'DP':
        leaves += [(150., 300., 'all', 'DP:intermediate')]
    if state != 'DD':
        leaves += [(300., finalize_sideband_scan_validation_np.inf, 'all', state + ':high')]
    if state == 'DD':
        result = dict(low)
        result['atoms'] = atoms.assign(mass_low=0., mass_high=150., prompt_group='all', leaf='DD:low')
        result['leaves'] = leaves
        return result
    value, variance, signal, signal_variance, atom_parts = [], [], [], [], []
    for lo, hi, prompt, name in leaves:
        value.append(finalize_sideband_scan_validation_previous._native_atomic_values(mc['backgrounds'][name], atoms, 'values'))
        variance.append(finalize_sideband_scan_validation_previous._native_atomic_values(mc['backgrounds'][name], atoms, 'variances'))
        signal.append(finalize_sideband_scan_validation_np.asarray([finalize_sideband_scan_validation_previous._native_atomic_values(mc['signals'][name][sample], atoms, 'values') for sample in names]))
        signal_variance.append(finalize_sideband_scan_validation_np.asarray([finalize_sideband_scan_validation_previous._native_atomic_values(mc['signals'][name][sample], atoms, 'variances') for sample in names]))
        atom_parts.append(atoms.assign(mass_low=lo, mass_high=hi, prompt_group=prompt, leaf=name))
    result = dict(atoms=finalize_sideband_scan_validation_pd.concat(atom_parts, ignore_index=True), values=finalize_sideband_scan_validation_np.concatenate(value),
        variances=finalize_sideband_scan_validation_np.concatenate(variance), signals=finalize_sideband_scan_validation_np.concatenate(signal, axis=1),
        signal_variances=finalize_sideband_scan_validation_np.concatenate(signal_variance, axis=1), signal_names=names, leaves=leaves)
    for field in ('values', 'variances', 'signals', 'signal_variances'):
        expected = low[field]
        actual = result[field][:, :len(atoms)] if field.startswith('signal') else result[field][:len(atoms)]
        finalize_sideband_scan_validation_np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
        if not finalize_sideband_scan_validation_np.isfinite(result[field]).all() or finalize_sideband_scan_validation_np.any(result[field] < -1e-8):
            raise AssertionError('Joint physical MC atoms must be nonnegative and finite.')
    for mass, lower, upper in [('intermediate', 150., 300.), ('all', 0., finalize_sideband_scan_validation_np.inf)]:
        mask = ((result['atoms'].mass_low >= lower) & (result['atoms'].mass_high <= upper)).to_numpy()
        nleaf = int(mask.sum()) // len(atoms)
        for field in ('values', 'variances'):
            actual = result[field][mask].reshape(nleaf, len(atoms)).sum(axis=0)
            expected = finalize_sideband_scan_validation_previous._native_atomic_values(mc['backgrounds'][state + ':' + mass], atoms, field)
            finalize_sideband_scan_validation_np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
        for sample_index, sample in enumerate(names):
            for field, native in [('signals', 'values'), ('signal_variances', 'variances')]:
                actual = result[field][sample_index, mask].reshape(nleaf, len(atoms)).sum(axis=0)
                expected = finalize_sideband_scan_validation_previous._native_atomic_values(mc['signals'][state + ':' + mass][sample], atoms, native)
                finalize_sideband_scan_validation_np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    validation.append(dict(scope=state, check='Joint native mass/prompt atoms reproduce coarse low/intermediate/all MC and every signal hypothesis', passed=True))
    return result


def finalize_sideband_scan_validation__scope_masks(spec, atomic, mass, bounds=None, prompt='all'):
    atoms = atomic['atoms']
    low, high = bounds if bounds is not None else {'low': (0., 150.), 'intermediate': (150., 300.), 'all': (0., finalize_sideband_scan_validation_np.inf)}[mass]
    mass_mask = ((atoms.mass_low >= low) & (atoms.mass_high <= high)).to_numpy()
    if prompt != 'all':
        mass_mask &= atoms.prompt_group.eq(prompt).to_numpy()
    return {region: mask & mass_mask for region, mask in finalize_sideband_scan_validation_previous.candidate_masks(spec, atoms).items()}


def finalize_sideband_scan_validation__unique_masks(masks):
    result, seen = {}, set()
    for name, mask in masks.items():
        token = finalize_sideband_scan_validation_np.packbits(mask).tobytes()
        if token not in seen:
            result[name] = mask
            seen.add(token)
    return result


def finalize_sideband_scan_validation__prior_masks(historical, last, state, atomic):
    result = {}
    source = historical['report_lowmass_dd'] if state == 'DD' else historical['mass_closure']
    selected = source.loc[source.source.eq('DATA') & source.display_allowed & source.plane.eq(state)]
    for row in selected.itertuples():
        spec = dict(a_iso=row.tx, b_iso=row.tx, a_phi=row.ty, c_phi=row.ty)
        mass = 'low' if state == 'DD' else row.mass_slice
        for r, mask in finalize_sideband_scan_validation__scope_masks(spec, atomic, mass).items():
            result[f'archive:{state}:{mass}:{row.wp}:{r}'] = mask
    strata = historical['strata_closure']
    for row in strata.loc[strata.source.eq('DATA') & strata.display_allowed & strata.plane.eq(state)].itertuples():
        spec = dict(a_iso=row.tx, b_iso=row.tx, a_phi=row.ty, c_phi=row.ty)
        masks = finalize_sideband_scan_validation__scope_masks(spec, atomic, row.mass_slice, (row.mass_low, row.mass_high), row.prompt_group)
        if not any(mask.any() for mask in masks.values()):
            raise AssertionError('A prior fine DATA selection has no representing mass/prompt atoms.')
        for r, mask in masks.items():
            result[f'archive:fine:{row.selection_id}:{row.wp}:{r}'] = mask
    prior = last['sbvr_closure']
    for row in prior.loc[prior.source.eq('DATA') & prior.display_allowed & prior.plane.eq(state)].to_dict('records'):
        for r, mask in finalize_sideband_scan_validation__scope_masks(row, atomic, row['mass_slice']).items():
            result[f'last:{row["scope"]}:{row["candidate_id"]}:{r}'] = mask
    # Previous nominal BCD observations are DD>=150.  They have no overlap with
    # this helper's only DD scope (MJJ<150), or with PP/PD/DP at any mass.
    nominal = last['sbd_regions'].loc[last['sbd_regions'].display_allowed]
    if not set(nominal.region).issubset(set('BCD')) or nominal.nominal_A_evaluated.any():
        raise AssertionError('The prior nominal release violates its BCD-only policy.')
    return finalize_sideband_scan_validation__unique_masks(result)


def finalize_sideband_scan_validation__cached_sources(caches, mass, state, spec):
    if mass == 'all':
        return True, None
    grid = caches[mass]['grid']
    frame = grid.loc[grid.source.eq('DATA') & grid.plane.eq(state)]
    rows = {}
    for region, (x, y) in finalize_sideband_scan_validation_candidate_coordinates(spec).items():
        found = frame.loc[finalize_sideband_scan_validation_np.isclose(frame.tx, x) & finalize_sideband_scan_validation_np.isclose(frame.ty, y)]
        if len(found) != 1:
            raise AssertionError('Every requested source coordinate must be unique.')
        rows[region] = found.iloc[0]
    return bool(all(row.approved for row in rows.values())), rows


def finalize_sideband_scan_validation__regress_candidate_mc(caches, mass, state, spec, values, variances, validation):
    frame = caches[mass]['grid'].query("source == 'MC' and plane == @state")
    for region, (x, y) in finalize_sideband_scan_validation_candidate_coordinates(spec).items():
        expected = frame.loc[finalize_sideband_scan_validation_np.isclose(frame.tx, x) & finalize_sideband_scan_validation_np.isclose(frame.ty, y)].iloc[0]
        finalize_sideband_scan_validation_np.testing.assert_allclose([values[region], variances[region]],
            [expected[region], expected['var_' + region]], rtol=1e-8, atol=1e-8)
    validation.append(dict(scope=mass + '_' + state + ':' + spec['candidate_id'],
        check='Native joint-atom ABCD moments reproduce original source cache rectangles', passed=True))


def finalize_sideband_scan_validation__load_inclusive_data(config, approved):
    """Query only prospectively approved all-mass PP/PD/DP ABCD rectangles."""
    bkg = finalize_sideband_scan_validation_Path(config['bkg_dir'])
    directory = bkg.with_name(bkg.name.replace('_bkg_', '_data_'))
    paths = sorted(directory.glob('*.coffea'))
    source = finalize_sideband_scan_validation_pd.read_csv(finalize_sideband_scan_validation_Path(config['vr2_output_dir']) / 'input_manifest.csv')
    source = source.loc[source.source.eq('DATA selected VR candidates')].set_index('path')
    if len(paths) != 68 or set(map(str, paths)) != set(source.index):
        raise AssertionError('Require the same 68 established original DATA files.')
    totals = {(spec['plane'], spec['candidate_id']): {r: 0. for r in finalize_sideband_scan_validation_REGIONS} for spec in approved}
    manifest = []
    for index, path in enumerate(paths, 1):
        record = finalize_sideband_scan_validation_common.candidate_input_record(path, 'Joint-audit-approved inclusive PP/PD/DP ABCD DATA slices')
        expected = source.loc[str(path)]
        if (record['sha256'], record['size_bytes']) != (expected.sha256, expected.size_bytes):
            raise AssertionError('Original DATA file changed since the established input manifest.')
        loaded, payload = finalize_sideband_scan_validation_common._extension_data_payload(path)
        weights, variances, edges = finalize_sideband_scan_validation_common._comparison_current_cube(payload, path, config)
        for spec in approved:
            if spec['plane'] not in ('PP', 'PD', 'DP') or spec['mass_slice'] != 'all':
                raise AssertionError('Only whitelisted all-mass P-containing VR DATA may be queried.')
            base = list(finalize_sideband_scan_validation__state_index(edges, spec['plane'], config))
            base.append(slice(finalize_sideband_scan_validation_common._comparison_split(edges['ljlj_mass'], 0.), None))
            for region, (x, y) in finalize_sideband_scan_validation_candidate_coordinates(spec).items():
                selection = list(base)
                ix = finalize_sideband_scan_validation_common._comparison_split(edges['iso0'], x)
                iy = finalize_sideband_scan_validation_common._comparison_split(edges['abs_dphi'], y)
                zero = finalize_sideband_scan_validation_common._comparison_split(edges['iso0'], 0.)
                selection[0] = slice(zero, ix) if region in 'AC' else slice(ix, None)
                selection[4] = slice(iy, None) if region in 'AB' else slice(None, iy)
                values, vs = weights[tuple(selection)], variances[tuple(selection)]
                if (not finalize_sideband_scan_validation_np.isfinite(values).all() or (values < 0).any()
                        or not finalize_sideband_scan_validation_np.array_equal(values, vs) or not finalize_sideband_scan_validation_np.array_equal(values, finalize_sideband_scan_validation_np.rint(values))):
                    raise AssertionError('Approved DATA cells must have integer counts and sumw2=N.')
                totals[(spec['plane'], spec['candidate_id'])][region] += float(values.sum())
        manifest.append(record)
        del loaded, payload, weights, variances
        finalize_sideband_scan_validation_gc.collect()
        if index % 20 == 0 or index == len(paths):
            print(f'09 approved inclusive VR DATA {index}/{len(paths)}', flush=True)
    return totals, manifest


def finalize_sideband_scan_validation__regress_previous_data(closure, last, validation):
    old = last['sbvr_closure'].query("source == 'DATA' and display_allowed")
    new = closure.query("source == 'DATA'").set_index(['scope', 'candidate_id'])
    fields = finalize_sideband_scan_validation_previous.MEASUREMENT_FIELDS
    for row in old.to_dict('records'):
        actual = new.loc[(row['scope'], row['candidate_id'])]
        if not actual.display_allowed:
            raise AssertionError('A previously displayed DATA observation was unexpectedly withheld.')
        finalize_sideband_scan_validation_np.testing.assert_allclose(actual[fields].to_numpy(float), [row[field] for field in fields],
                                   rtol=1e-8, atol=1e-8, equal_nan=True)
    validation.append(dict(scope='Previous release', check='Every one of the 34 previously displayed sideband VR observations is reproduced unchanged', passed=True))


def finalize_sideband_scan_validation__validate_prior_fine_mc(mc, historical, validation):
    for row in historical['strata_closure'].query("source == 'MC'").itertuples():
        actual = finalize_sideband_scan_validation_common._comparison_regions(mc['backgrounds']['fine:' + row.selection_id], row.tx, row.ty)
        for region in finalize_sideband_scan_validation_REGIONS:
            finalize_sideband_scan_validation_np.testing.assert_allclose([actual[region]['value'], actual[region]['variance']],
                [getattr(row, region), getattr(row, 'var_' + region)], rtol=1e-8, atol=1e-8)
    for row in historical['strata_signal_points'].itertuples():
        actual = finalize_sideband_scan_validation_common._comparison_regions(mc['signals']['fine:' + row.selection_id][row.sample], row.tx, row.ty)
        for region in finalize_sideband_scan_validation_REGIONS:
            finalize_sideband_scan_validation_np.testing.assert_allclose([actual[region]['value'], actual[region]['variance']],
                [getattr(row, 'S_' + region), getattr(row, 'var_S_' + region)], rtol=1e-8, atol=1e-8)
    validation.append(dict(scope='Prior fine MC',
        check='Every original mass/prompt leaf reproduces archived background and all-120-signal moments at Ref/WP1/WP2', passed=True))


def finalize_sideband_scan_validation__validate_joint_complement_safety(validation):
    old = {'low': finalize_sideband_scan_validation_np.array([1, 0, 0], bool), 'intermediate': finalize_sideband_scan_validation_np.array([0, 1, 0], bool)}
    new = {'inclusive': finalize_sideband_scan_validation_np.ones(3, bool)}
    for high_background, high_signal, expected in [(1., .2, False), (0., 0., False), (10., .2, True)]:
        signal = finalize_sideband_scan_validation_np.zeros((120, 3))
        signal[0, 2] = high_signal
        atomic = dict(values=finalize_sideband_scan_validation_np.array([100., 100., high_background]),
                      variances=finalize_sideband_scan_validation_np.array([100., 100., high_background]),
                      signals=signal, signal_names=[f'model_{i}' for i in range(120)])
        audit = finalize_sideband_scan_validation_joint_disclosure_audit(old, new, atomic)
        if not len(audit) or bool(audit.approved.all()) != expected:
            raise AssertionError('The mass-inclusive gate failed its unsafe/unsupported-high-remainder test.')
    validation.append(dict(scope='Joint-mass safety',
        check='Synthetic inclusive observations with safe totals reject contaminated or unsupported high remainders and accept a safe remainder', passed=True))


def finalize_sideband_scan_validation_joint_disclosure_audit(old_masks, new_masks, atomic):
    """Batch the established linear-disclosure audit without old-only pairs.

    A difference of two prior measurements is already known, so it cannot be
    newly disclosed. Removing those pairs avoids repeating thousands of SVD
    projections at every scan point while preserving the original criterion.
    """
    width = len(atomic['values'])
    combined = {**old_masks, **new_masks}
    labels = list(combined)
    rows = finalize_sideband_scan_validation_np.asarray(list(combined.values()), bool)
    new_names = set(new_masks)
    patterns = {}
    def add(mask, kind, description):
        if not finalize_sideband_scan_validation_np.any(mask):
            return
        token = finalize_sideband_scan_validation_np.packbits(mask).tobytes()
        patterns.setdefault(token, dict(mask=mask, audit_kind=kind, description=description))
    if len(rows):
        _, inverse = finalize_sideband_scan_validation_np.unique(rows.T, axis=0, return_inverse=True)
        for group in finalize_sideband_scan_validation_np.unique(inverse):
            add(inverse == group, 'IDENTIFIABLE_PARTITION_BLOCK', f'membership group {int(group)}')
        for name, mask in new_masks.items():
            add(mask, 'NEW_REGION', name)
        for i, outer in enumerate(rows):
            contained = []
            for j, inner in enumerate(rows):
                if i == j or not inner.any() or not finalize_sideband_scan_validation_np.all(~inner | outer):
                    continue
                if labels[i] in new_names or labels[j] in new_names:
                    add(outer & ~inner, 'NESTED_REGION_COMPLEMENT', labels[i] + ' minus ' + labels[j])
                if not finalize_sideband_scan_validation_np.array_equal(inner, outer):
                    contained.append((labels[j], inner))
            # Prior-only remainders are already in the old row space.
            if labels[i] not in new_names and not any(name in new_names for name, _ in contained):
                continue
            used, removed = finalize_sideband_scan_validation_np.zeros(width, bool), []
            for name, inner in sorted(contained, key=lambda item: -int(item[1].sum())):
                if not (used & inner).any():
                    used |= inner
                    removed.append(name)
            if removed:
                add(outer & ~used, 'DISJOINT_SUBSET_REMAINDER', labels[i] + ' minus ' + ','.join(removed))
    columns = ['audit_kind', 'description', 'atomic_cells', 'background', 'var_background',
               'max_S_B', 'worst_sample', 'signal_hypotheses', 'approved', 'reason']
    if not patterns:
        return finalize_sideband_scan_validation_pd.DataFrame(columns=columns)
    records = list(patterns.values())
    matrix = finalize_sideband_scan_validation_np.asarray([record['mask'] for record in records], float)
    old_basis = finalize_sideband_scan_validation_previous._basis(old_masks.values(), width)
    basis = finalize_sideband_scan_validation_previous._basis(combined.values(), width)
    with finalize_sideband_scan_validation_threadpool_limits(limits=1, user_api='blas'):
        old_known = finalize_sideband_scan_validation_np.max(finalize_sideband_scan_validation_np.abs(matrix - (matrix @ old_basis.T) @ old_basis), axis=1) < 1e-8
        new_known = finalize_sideband_scan_validation_np.max(finalize_sideband_scan_validation_np.abs(matrix - (matrix @ basis.T) @ basis), axis=1) < 1e-8
    output = []
    for record, keep in zip(records, ~old_known & new_known):
        if not keep:
            continue
        mask = record.pop('mask')
        background = float(atomic['values'][mask].sum())
        variance = float(atomic['variances'][mask].sum())
        signal = atomic['signals'][:, mask].sum(axis=1)
        ratios = finalize_sideband_scan_validation_previous._ratios(signal, background)
        valid = bool(background > 1e-8 and finalize_sideband_scan_validation_np.isfinite(ratios).all())
        worst = int(finalize_sideband_scan_validation_np.argmax(signal))
        maximum = float(ratios[worst]) if valid else finalize_sideband_scan_validation_np.nan
        output.append(dict(**record, atomic_cells=int(mask.sum()), background=background,
            var_background=variance, max_S_B=maximum, worst_sample=atomic['signal_names'][worst],
            signal_hypotheses=120, approved=bool(valid and maximum < .1),
            reason='PASS' if valid and maximum < .1 else 'NONPOSITIVE_BACKGROUND' if not valid
            else 'CONTAMINATION_AT_OR_ABOVE_10_PERCENT'))
    return finalize_sideband_scan_validation_pd.DataFrame(output, columns=columns)


def finalize_sideband_scan_validation__exhaustive_guard(atomic, published, x_edges, y_edges):
    """Precompute every unsafe rectangle times every nonempty leaf union.

    The separable representation avoids storing a large dense matrix for all
    mass-subset masks.  Only the remaining squared projection distance to the
    current public row space is updated when an observation is accepted.
    """
    atoms = atomic['atoms']
    leaves = atoms.leaf.drop_duplicates().tolist()
    base = atoms.loc[atoms.leaf.eq(leaves[0])]
    nbase, nleaf = len(base), len(leaves)
    rectangles, coordinates = [], []
    for lx in range(len(x_edges)-1):
        for hx in range(lx+1, len(x_edges)):
            for ly in range(len(y_edges)-1):
                for hy in range(ly+1, len(y_edges)):
                    mask = ((base.xlow >= x_edges[lx]) & (base.xhigh <= x_edges[hx])
                            & (base.ylow >= y_edges[ly]) & (base.yhigh <= y_edges[hy])).to_numpy(bool)
                    rectangles.append(mask)
                    coordinates.append((x_edges[lx], x_edges[hx], y_edges[ly], y_edges[hy]))
    rectangles = finalize_sideband_scan_validation_np.asarray(rectangles, float)
    subsets = finalize_sideband_scan_validation_np.asarray([[(bits >> j) & 1 for j in range(nleaf)]
                          for bits in range(1, 1 << nleaf)], bool)
    values = atomic['values'].reshape(nleaf, nbase)
    variances = atomic['variances'].reshape(nleaf, nbase)
    signals = atomic['signals'].reshape(120, nleaf, nbase)
    subset_indices, rectangle_indices, backgrounds, background_variances, maxima, worsts = [], [], [], [], [], []
    with finalize_sideband_scan_validation_threadpool_limits(limits=1, user_api='blas'):
        for subset_index, subset in enumerate(subsets):
            b = rectangles @ values[subset].sum(axis=0)
            variance = rectangles @ variances[subset].sum(axis=0)
            s = signals[:, subset].sum(axis=1) @ rectangles.T
            worst = finalize_sideband_scan_validation_np.argmax(s, axis=0)
            maximum_signal = s[worst, finalize_sideband_scan_validation_np.arange(len(rectangles))]
            ratios = finalize_sideband_scan_validation_np.full(len(b), finalize_sideband_scan_validation_np.nan)
            positive = b > 1e-8
            ratios[positive] = maximum_signal[positive] / b[positive]
            bad = ~positive | ~finalize_sideband_scan_validation_np.isfinite(ratios) | (ratios >= .1)
            indices = finalize_sideband_scan_validation_np.flatnonzero(bad)
            subset_indices.extend([subset_index] * len(indices))
            rectangle_indices.extend(indices.tolist())
            backgrounds.extend(b[indices].tolist())
            background_variances.extend(variance[indices].tolist())
            maxima.extend(ratios[indices].tolist())
            worsts.extend(worst[indices].tolist())
    guard = dict(rectangles=rectangles, coordinates=coordinates, subsets=subsets,
        unsafe_subset=finalize_sideband_scan_validation_np.asarray(subset_indices, int), unsafe_rectangle=finalize_sideband_scan_validation_np.asarray(rectangle_indices, int),
        backgrounds=finalize_sideband_scan_validation_np.asarray(backgrounds), background_variances=finalize_sideband_scan_validation_np.asarray(background_variances),
        maxima=finalize_sideband_scan_validation_np.asarray(maxima), worsts=finalize_sideband_scan_validation_np.asarray(worsts, int), leaves=leaves,
        nbase=nbase, nleaf=nleaf, signal_names=atomic['signal_names'],
        basis=finalize_sideband_scan_validation_previous._basis(published.values(), len(atomic['values'])),
        total_rectangles=len(rectangles)*len(subsets), total_unsafe=len(rectangle_indices),
        probes=0, rejected=0)
    norms = rectangles.sum(axis=1)[guard['unsafe_rectangle']] * subsets.sum(axis=1)[guard['unsafe_subset']]
    projected = finalize_sideband_scan_validation__guard_projection_squared(guard, guard['basis'])
    residual = norms - projected
    old_known = finalize_sideband_scan_validation_np.abs(residual) < 1e-7
    guard['grandfathered'] = int(old_known.sum())
    for key in ('unsafe_subset', 'unsafe_rectangle', 'backgrounds', 'background_variances', 'maxima', 'worsts'):
        guard[key] = guard[key][~old_known]
    guard['residual'] = residual[~old_known]
    if finalize_sideband_scan_validation_np.any(guard['residual'] < -1e-7):
        raise AssertionError('Invalid initial projection norm for the exhaustive disclosure guard.')
    return guard


def finalize_sideband_scan_validation__guard_projection_squared(guard, basis):
    count = len(guard['unsafe_rectangle'])
    if not len(basis):
        return finalize_sideband_scan_validation_np.zeros(count)
    nbase = guard['nbase']
    with finalize_sideband_scan_validation_threadpool_limits(limits=1, user_api='blas'):
        coefficient = finalize_sideband_scan_validation_np.asarray([guard['rectangles'] @ basis[:, i*nbase:(i+1)*nbase].T
                                  for i in range(guard['nleaf'])])
        output = finalize_sideband_scan_validation_np.empty(count)
        for start in range(0, count, 2048):
            stop = min(start+2048, count)
            rect = guard['unsafe_rectangle'][start:stop]
            sub = guard['subsets'][guard['unsafe_subset'][start:stop]]
            projected = finalize_sideband_scan_validation_np.einsum('ul,lur->ur', sub, coefficient[:, rect, :])
            output[start:stop] = finalize_sideband_scan_validation_np.einsum('ij,ij->i', projected, projected)
    return output


def finalize_sideband_scan_validation__guard_probe(guard, proposed):
    matrix = finalize_sideband_scan_validation_np.asarray(list(proposed.values()), float)
    basis = guard['basis']
    with finalize_sideband_scan_validation_threadpool_limits(limits=1, user_api='blas'):
        residual = matrix - (matrix @ basis.T) @ basis
        _, singular, vh = finalize_sideband_scan_validation_np.linalg.svd(residual, full_matrices=False)
    rank = int(finalize_sideband_scan_validation_np.count_nonzero(singular > 1e-9))
    increment = vh[:rank]
    if rank and len(basis):
        finalize_sideband_scan_validation_np.testing.assert_allclose(increment @ basis.T, 0., rtol=0., atol=1e-8)
    decrease = finalize_sideband_scan_validation__guard_projection_squared(guard, increment)
    remaining = guard['residual'] - decrease
    bad = remaining < 1e-7
    guard['probes'] += 1
    guard['rejected'] += int(bad.any())
    rows = []
    for index in finalize_sideband_scan_validation_np.flatnonzero(bad):
        rectangle = guard['unsafe_rectangle'][index]
        subset = guard['subsets'][guard['unsafe_subset'][index]]
        selected_leaves = [name for name, selected in zip(guard['leaves'], subset) if selected]
        xlo, xhi, ylo, yhi = guard['coordinates'][rectangle]
        b = guard['backgrounds'][index]
        rows.append(dict(audit_kind='EXHAUSTIVE_NATIVE_RECTANGLE_MASS_SUBSET',
            description=f'iso[{xlo:g},{xhi:g}), phi[{ylo:g},{yhi:g}); leaves=' + ','.join(selected_leaves),
            atomic_cells=int(guard['rectangles'][rectangle].sum() * subset.sum()),
            background=b, var_background=guard['background_variances'][index],
            max_S_B=guard['maxima'][index], worst_sample=guard['signal_names'][guard['worsts'][index]],
            signal_hypotheses=120, approved=False,
            reason='NONPOSITIVE_BACKGROUND' if b <= 1e-8 else 'CONTAMINATION_AT_OR_ABOVE_10_PERCENT'))
    return dict(audit=finalize_sideband_scan_validation_pd.DataFrame(rows), increment=increment, decrease=decrease)


def finalize_sideband_scan_validation__guard_commit(guard, probe):
    if probe is None or len(probe['audit']):
        raise AssertionError('Only an exhaustively approved query may update the public row space.')
    guard['basis'] = finalize_sideband_scan_validation_np.vstack([guard['basis'], probe['increment']])
    guard['residual'] -= probe['decrease']
    if finalize_sideband_scan_validation_np.any(guard['residual'] < 1e-7):
        raise AssertionError('An unsafe rectangle became identifiable after accepting a query.')


def finalize_sideband_scan_validation_data_statistics(values, variances, min_neff=10.):
    """Count diagnostics with undefined symmetric error for an empty numerator."""
    counts = finalize_sideband_scan_validation_np.asarray([values[r] for r in finalize_sideband_scan_validation_REGIONS], float)
    variance = finalize_sideband_scan_validation_np.asarray([variances[r] for r in finalize_sideband_scan_validation_REGIONS], float)
    if (not finalize_sideband_scan_validation_np.isfinite(counts).all() or (counts < 0).any()
            or not finalize_sideband_scan_validation_np.array_equal(counts, finalize_sideband_scan_validation_np.rint(counts)) or not finalize_sideband_scan_validation_np.array_equal(counts, variance)):
        raise AssertionError('Approved VR DATA must be nonnegative integer counts with variance=N.')
    result = finalize_sideband_scan_validation_statistics_from_regions(values, variances, min_neff)
    if values['D'] > 0 and (values['B'] == 0 or values['C'] == 0):
        result.update(pred_error=finalize_sideband_scan_validation_np.nan, status='ZERO_NUMERATOR_INTERVAL_REQUIRED', low_stat=True)
    return result


def finalize_sideband_scan_validation__validate_final_exhaustive_basis(guard, published):
    """Recompute the full final row space with QR rather than incremental SVD."""
    from scipy.linalg import qr as qr
    matrix = finalize_sideband_scan_validation_np.asarray(list(published.values()), float).T
    with finalize_sideband_scan_validation_threadpool_limits(limits=1, user_api='blas'):
        q, r, _ = qr(matrix, mode='economic', pivoting=True)
    rank = int(finalize_sideband_scan_validation_np.count_nonzero(finalize_sideband_scan_validation_np.abs(finalize_sideband_scan_validation_np.diag(r)) > 1e-9))
    basis = q[:, :rank].T
    norm = (guard['rectangles'].sum(axis=1)[guard['unsafe_rectangle']]
            * guard['subsets'].sum(axis=1)[guard['unsafe_subset']])
    residual = norm - finalize_sideband_scan_validation__guard_projection_squared(guard, basis)
    finalize_sideband_scan_validation_np.testing.assert_allclose(residual, guard['residual'], rtol=0., atol=1e-7)
    unsafe = int((finalize_sideband_scan_validation_np.abs(residual) < 1e-7).sum())
    if unsafe:
        raise AssertionError('Independent QR recomputation resolves an unsafe native rectangle.')
    return unsafe


# ========================================================================
# abcd_buffer_distributions: original study API
# ========================================================================
"""Buffered 2D MC views and strictly pre-existing, gated VR DATA display bins.

The displayed bins use the established 01/03 tail/flow folding.  DATA are
limited to the selected buffer's allowed VRs and to precisely the same safe
bins already displayed in 01/03.  Exact query-mask reuse proves that no new
linear information is added to any combination of previous disclosures.
No nominal DATA region or complete DATA plane is selected or projected.
"""
from pathlib import Path as finalize_buffer_distributions_Path
import gc as finalize_buffer_distributions_gc
import hashlib as finalize_buffer_distributions_hashlib
import json as finalize_buffer_distributions_json

import numpy as finalize_buffer_distributions_np
import pandas as finalize_buffer_distributions_pd

finalize_buffer_distributions_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_buffer_distributions_scan_validation = _FINALIZE_NAMESPACES['abcd_sideband_scan_validation']
finalize_buffer_distributions_sideband_validation = _FINALIZE_NAMESPACES['abcd_sideband_validation']
finalize_buffer_distributions_candidate_coordinates = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].candidate_coordinates
finalize_buffer_distributions__opt_distribution_edges = _opt_distribution_edges; finalize_buffer_distributions__opt_fold_distribution = _opt_fold_distribution
finalize_buffer_distributions_strata_specs = _FINALIZE_NAMESPACES['abcd_final_check_strata']._specs; finalize_buffer_distributions_project_strata = _FINALIZE_NAMESPACES['abcd_final_check_strata']._project_mc
finalize_buffer_distributions_project_mass_partitions = _FINALIZE_NAMESPACES['abcd_mass_inclusive'].project_mass_partitions; finalize_buffer_distributions__state_index = _FINALIZE_NAMESPACES['abcd_mass_inclusive']._state_index


finalize_buffer_distributions_BUFFER = dict(candidate_id='G_i0p3_p2', a_iso=.2, a_phi=2.7, b_iso=.3, c_phi=2.)
finalize_buffer_distributions_SCOPES = [('Nominal_DD', 'nominal', 'DD'), ('VR1_DD', 'low', 'DD'),
          ('VR1_PP', 'low', 'PP'), ('VR1_PD', 'low', 'PD'), ('VR1_DP', 'low', 'DP'),
          ('VR2_PP', 'intermediate', 'PP'), ('VR2_PD', 'intermediate', 'PD'),
          ('VR2_DP', 'intermediate', 'DP'), ('INCLUSIVE_PP', 'all', 'PP'),
          ('INCLUSIVE_PD', 'all', 'PD'), ('INCLUSIVE_DP', 'all', 'DP')]
finalize_buffer_distributions_ALLOWED_DATA_SCOPES = {'VR1_PP', 'VR1_PD', 'VR2_PP', 'VR2_PD'}
finalize_buffer_distributions_FLOW_NOTE = ('Display bins are event sums, not densities. Isolation underflow is excluded; '
             'the final displayed isolation bin [1,2] includes the physical interval [1,infinity). '
             'Angle underflow/overflow are folded into the first/last displayed angle bins. '
             'The MC, signal-purity and DATA query use these identical physical bin definitions.')


def finalize_buffer_distributions_run_buffer_distributions(config, historical, previous_sideband, latest_scan,
                             load_data=False, mc=None):
    """Build buffered MC planes and redisplay only previously published safe bins."""
    if (config['topology'], config['purity_limit'], config['partner_iso_low'],
            config['partner_iso_high']) != ('2mu2e', .1, 0., .5):
        raise ValueError('Require the established 2mu2e, strict 10%, partner [0,.5) conventions.')
    historical = historical.get('tables', historical)
    previous_sideband = previous_sideband.get('tables', previous_sideband)
    latest_scan = latest_scan.get('tables', latest_scan)
    inputs, validation = [], []
    finalize_buffer_distributions_scan_validation._audit_prior_snapshots(config, historical, previous_sideband, inputs, validation)
    finalize_buffer_distributions__audit_latest_snapshot(config, latest_scan, inputs, validation)
    prior_bins = finalize_buffer_distributions__display_priors(config, inputs, validation)
    supplied_mc = mc is not None
    mc = finalize_buffer_distributions_load_buffer_mc(config) if mc is None else mc
    inputs.extend(mc['manifest'])
    validation.extend(mc['validation'])
    if not all(name in mc['backgrounds'] for name in ('DD:nominal', 'DD:low')):
        raise ValueError('Buffer distribution MC must include the two MC-only DD planes.')
    reference = mc['backgrounds']['DD:nominal']
    xedges = finalize_buffer_distributions__opt_distribution_edges([reference], 'xedges', 2.)
    yedges = finalize_buffer_distributions__opt_distribution_edges([reference], 'yedges', finalize_buffer_distributions_np.pi)
    physical = finalize_buffer_distributions__physical_bins(xedges, yedges)
    dimensions = (len(xedges)-1, len(yedges)-1)
    current = latest_scan['sv_closure']
    gates = latest_scan['sv_release_audit'].loc[
        latest_scan['sv_release_audit'].candidate_id.eq(finalize_buffer_distributions_BUFFER['candidate_id'])].set_index('scope')
    allowed = set(gates.loc[gates.display_allowed, :].index)
    if allowed != finalize_buffer_distributions_ALLOWED_DATA_SCOPES:
        raise AssertionError('The selected buffer must retain the established four allowed VR scopes.')
    planes, bin_rows, scope_rows, proof_rows = {}, [], [], []
    publics = {}
    for state in ('DD', 'PP', 'PD', 'DP'):
        atomic = finalize_buffer_distributions__native_atomic(mc, state)
        published = finalize_buffer_distributions_scan_validation._prior_masks(historical, previous_sideband, state, atomic)
        already = current.loc[current.source.eq('DATA') & current.display_allowed & current.plane.eq(state)]
        for row in already.to_dict('records'):
            for region, mask in finalize_buffer_distributions_scan_validation._scope_masks(row, atomic, row['mass_slice']).items():
                published[f'latest:{row["scope"]}:{row["candidate_id"]}:{region}'] = mask
        for row in prior_bins.loc[prior_bins.plane.eq(state) & prior_bins.previously_displayed].to_dict('records'):
            published[f'old2D:{row["scope"]}:{row["ix"]}:{row["iy"]}'] = finalize_buffer_distributions__bin_mask(atomic, row, row['mass_slice'])
        published = finalize_buffer_distributions_scan_validation._unique_masks(published)
        publics[state] = dict(atomic=atomic, masks=published,
            tokens={finalize_buffer_distributions_np.packbits(mask).tobytes() for mask in published.values()})
    for scope, mass, state in finalize_buffer_distributions_SCOPES:
        key = state + ':' + mass
        background = mc['backgrounds'][key]
        signals = mc['signals'][key]
        names = sorted(signals)
        if len(names) != 120:
            raise AssertionError('Every distribution requires the same 120 separate signal hypotheses.')
        b = finalize_buffer_distributions__opt_fold_distribution(background, xedges, yedges, True)[0]
        vb = finalize_buffer_distributions__opt_fold_distribution(dict(background, values=background['variances']), xedges, yedges, True)[0]
        s = finalize_buffer_distributions_np.asarray([finalize_buffer_distributions__opt_fold_distribution(signals[name], xedges, yedges, True)[0] for name in names])
        valid_signal = finalize_buffer_distributions_np.isfinite(s).all(axis=0) & (s >= 0).all(axis=0)
        valid_background = finalize_buffer_distributions_np.isfinite(b) & (b > 1e-8)
        ratios = finalize_buffer_distributions_np.full(s.shape, finalize_buffer_distributions_np.nan)
        finalize_buffer_distributions_np.divide(s, b[None, :, :], out=ratios, where=valid_background[None, :, :])
        worst = finalize_buffer_distributions_np.argmax(s, axis=0)
        xx, yy = finalize_buffer_distributions_np.indices(dimensions)
        maximum = ratios[worst, xx, yy]
        pure = valid_background & valid_signal & finalize_buffer_distributions_np.isfinite(maximum) & (maximum < .1)
        integrated = bool(scope in allowed)
        reason = 'NOMINAL_DATA_ALWAYS_BLINDED' if mass == 'nominal' else str(gates.loc[scope, 'reason'])
        if mass != 'nominal':
            finalize_buffer_distributions__regress_buffer_mc(background, current, scope)
        else:
            finalize_buffer_distributions__regress_nominal_mc(background, latest_scan['sideband_scan'])
            validation.append(dict(scope=scope,
                check='Nominal DD MC buffered ABCD values and variances reproduce the established nominal cache', passed=True))
        old = prior_bins.loc[prior_bins.scope.eq(scope)].set_index(['ix', 'iy'])
        if mass in ('low', 'intermediate'):
            for cell in physical.to_dict('records'):
                ix, iy = cell['ix'], cell['iy']
                source = old.loc[(ix, iy)]
                finalize_buffer_distributions_np.testing.assert_allclose([b[ix, iy], vb[ix, iy], maximum[ix, iy]],
                    [source.background, source.background_variance, source.max_S_B],
                    rtol=1e-8, atol=1e-8, equal_nan=True)
                if bool(pure[ix, iy]) != bool(source.bin_approved):
                    raise AssertionError('The reconstructed exact display-bin purity differs from 01/03.')
            validation.append(dict(scope=scope,
                check='All 154 native display bins reproduce established folded MC yield, variance and 120-model purity', passed=True))
        visible = finalize_buffer_distributions_np.zeros(dimensions, bool)
        reasons = finalize_buffer_distributions_np.full(dimensions, reason, object)
        for cell in physical.to_dict('records'):
            ix, iy = cell['ix'], cell['iy']
            previous_display = bool(not old.empty and old.loc[(ix, iy)].previously_displayed)
            same_mask = False
            if integrated and pure[ix, iy] and previous_display:
                public = publics[state]
                proposed = finalize_buffer_distributions__bin_mask(public['atomic'], cell, mass)
                original = finalize_buffer_distributions__bin_mask(public['atomic'], old.loc[(ix, iy)].to_dict(), mass)
                if not finalize_buffer_distributions_np.array_equal(proposed, original):
                    raise AssertionError('A proposed DATA bin must equal the previously displayed physical mask exactly.')
                same_mask = finalize_buffer_distributions_np.packbits(proposed).tobytes() in public['tokens']
                if not same_mask:
                    raise AssertionError('A DATA query adds a new mask outside the pre-existing public set.')
                finalize_buffer_distributions_np.testing.assert_allclose(public['atomic']['values'][proposed].sum(), b[ix, iy], rtol=1e-10, atol=1e-8)
                proof_rows.append(dict(scope=scope, plane=state, mass_slice=mass, ix=ix, iy=iy,
                    audit_kind='EXACT_PREVIOUSLY_DISPLAYED_BIN', previous_public_masks=len(public['masks']),
                    exact_previous_mask=True, newly_released_linear_functionals=0,
                    newly_identifiable_unsafe_regions=0,
                    reason='Identical query already belongs to the public set; its row space is unchanged.'))
            eligible = bool(integrated and pure[ix, iy] and previous_display and same_mask)
            why = ('PREVIOUS_DISPLAY_BIN_PURITY_AND_CURRENT_BUFFER_GATE_PASS' if eligible else
                   'NOMINAL_DATA_ALWAYS_BLINDED' if mass == 'nominal' else
                   'SELECTED_BUFFER_VR_GATE_BLOCKED' if not integrated else
                   'BIN_BACKGROUND_UNSUPPORTED' if not valid_background[ix, iy] else
                   'BIN_SIGNAL_CONTAMINATION_OR_INVALID_SIGNAL' if not pure[ix, iy] else
                   'NOT_A_PREVIOUSLY_DISPLAYED_IDENTICAL_BIN')
            visible[ix, iy] = eligible
            reasons[ix, iy] = why
            bin_rows.append(dict(scope=scope, plane=state, mass_slice=mass, **cell,
                background=b[ix, iy], var_background=vb[ix, iy], max_S_B=maximum[ix, iy],
                worst_sample=names[worst[ix, iy]], signal_hypotheses=120,
                bin_purity_pass=bool(pure[ix, iy]), integrated_allowed=integrated,
                previously_displayed=previous_display, complementary_pass=same_mask,
                release_eligible=eligible, display_allowed=bool(eligible and load_data),
                reason=why if not eligible or load_data else 'DATA_EVALUATION_DISABLED',
                data=finalize_buffer_distributions_np.nan, var_data=finalize_buffer_distributions_np.nan))
        planes[scope] = dict(xedges=xedges.copy(), yedges=yedges.copy(), MC=b, DATA=finalize_buffer_distributions_np.full(dimensions, finalize_buffer_distributions_np.nan),
            variance_MC=vb, variance_DATA=finalize_buffer_distributions_np.full(dimensions, finalize_buffer_distributions_np.nan), approved=visible.copy(),
            display_allowed=visible.copy() if load_data else finalize_buffer_distributions_np.zeros(dimensions, bool),
            max_S_B=maximum, reason=reasons, plane=state, mass_slice=mass,
            integrated_allowed=integrated, integrated_reason=reason, flow_note=finalize_buffer_distributions_FLOW_NOTE)
        scope_rows.append(dict(scope=scope, plane=state, mass_slice=mass, integrated_allowed=integrated,
            integrated_reason=reason, display_bins=int(b.size), bin_purity_pass=int(pure.sum()),
            previously_displayed_bins=int(old.previously_displayed.sum()) if not old.empty else 0,
            release_eligible_bins=int(visible.sum()), display_allowed_bins=int(visible.sum()) if load_data else 0,
            nominal_A_evaluated=False))
    bins = finalize_buffer_distributions_pd.DataFrame(bin_rows)
    selected_bins = bins.loc[bins.release_eligible].copy()
    if len(selected_bins) != 502 or set(selected_bins.scope) != finalize_buffer_distributions_ALLOWED_DATA_SCOPES:
        raise AssertionError('Exactly the established 502 safe bins in four allowed VRs must be reused.')
    data, data_manifest = ({}, [])
    if load_data:
        data, data_manifest = finalize_buffer_distributions__load_data_bins(config, selected_bins)
        inputs.extend(data_manifest)
        for index, row in bins.loc[bins.display_allowed].iterrows():
            value = data[(row.scope, int(row.ix), int(row.iy))]
            bins.loc[index, ['data', 'var_data']] = value
            planes[row.scope]['DATA'][int(row.ix), int(row.iy)] = value
            planes[row.scope]['variance_DATA'][int(row.ix), int(row.iy)] = value
        finalize_buffer_distributions__validate_visible_counts(bins, current, validation)
    if not bins.loc[~bins.display_allowed, ['data', 'var_data']].isna().all().all():
        raise AssertionError('Every withheld DATA bin must remain NaN.')
    validation.extend([
        dict(scope='Display geometry', check='Identical exact tail/flow folding is used for background MC, all 120 signals and every DATA query', passed=True),
        dict(scope='Prior information', check='Every proposed DATA mask exactly equals an already displayed 01/03 bin; no new public linear functional exists', passed=True),
        dict(scope='Joint protection', check='Exact existing-mask reuse leaves all prior-integral, mass/prompt and nominal-sideband information unchanged', passed=True),
        dict(scope='Blinding', check='No nominal DATA or disallowed VR DATA is selected; all withheld values and variances are NaN', passed=True),
    ])
    return dict(planes=planes, tables=dict(bd_bins=bins, bd_scope_audit=finalize_buffer_distributions_pd.DataFrame(scope_rows),
        bd_disclosure_audit=finalize_buffer_distributions_pd.DataFrame(proof_rows), bd_validation=finalize_buffer_distributions_pd.DataFrame(validation),
        bd_input_manifest=finalize_buffer_distributions_pd.DataFrame(inputs).drop_duplicates(['path', 'sha256'])),
        metadata=dict(buffer=finalize_buffer_distributions_BUFFER.copy(), nominal_A_evaluated=False, nominal_BCD_evaluated=False,
            original_data_files_opened=len(data_manifest), original_mc_files_opened=0 if supplied_mc else 138,
            allowed_data_scopes=sorted(finalize_buffer_distributions_ALLOWED_DATA_SCOPES), eligible_data_bins=len(selected_bins),
            new_public_linear_functionals=0, newly_identifiable_unsafe_regions=0,
            signal_hypotheses=120, exact_previous_bin_reuse=True, load_data=bool(load_data), flow_note=finalize_buffer_distributions_FLOW_NOTE,
            protection='Selected-buffer integrated gate AND exact displayed-bin S/B_MC<10% over all 120 hypotheses AND exact reuse of a previously displayed physical bin. No new query mask is introduced.',
            row_space_proof='Each candidate query equals a member of the existing public observation set, so appending any or all of them leaves its span unchanged; this applies to every linear reconstruction, including every native rectangle and mass/prompt union.',
            data_policy='After MC-only authorization, re-read only the 502 approved VR bin slices from the identical 68 DATA files. No complete DATA projection, full-plane total or nominal region is evaluated.'))


def finalize_buffer_distributions__audit_latest_snapshot(config, latest, inputs, validation):
    base = finalize_buffer_distributions_Path(config['notebook_path']).parent
    directory = finalize_buffer_distributions_Path(config.get('buffer_reference_dir', base/'09_20261027_iso_reference'))
    manifest = directory/'archive_manifest.json'
    document = finalize_buffer_distributions_json.loads(manifest.read_text())
    for relative, digest in document['files'].items():
        if finalize_buffer_distributions_hashlib.sha256((directory/relative).read_bytes()).hexdigest() != digest:
            raise AssertionError('The preceding complete scan snapshot changed: ' + relative)
    inputs.append(finalize_buffer_distributions_common.candidate_input_record(manifest, 'Immutable preceding 09 scan and DATA disclosures'))
    for name in ('sv_closure', 'sv_release_audit'):
        path = directory/'09_2mu2e_final_check_outputs'/(name+'.csv')
        expected = finalize_buffer_distributions_pd.read_csv(path, float_precision='round_trip')
        finalize_buffer_distributions_pd.testing.assert_frame_equal(latest[name].reset_index(drop=True), expected, check_dtype=False)
        inputs.append(finalize_buffer_distributions_common.candidate_input_record(path, 'Previously displayed 261 VR combinations'))
    if int(latest['sv_release_audit'].display_allowed.sum()) != 261:
        raise AssertionError('Require the exact established 261-combination release set.')
    validation.append(dict(scope='Latest reference', check='The preceding 261-combination scan matches its immutable input manifest', passed=True))


def finalize_buffer_distributions__display_priors(config, inputs, validation):
    base = finalize_buffer_distributions_Path(config['notebook_path']).parent
    rows = []
    for folder, mass, prefix, bounds in [('01_2mu2e_VR1_summary_outputs', 'low', 'VR1', (0., 150.)),
                                         ('03_2mu2e_VR2_summary_outputs', 'intermediate', 'VR2', (150., 300.))]:
        directory = base/folder
        document = finalize_buffer_distributions_json.loads((directory/'manifest.json').read_text())
        settings = document['settings']
        if (settings['topology'], settings['mass_min'], settings['mass_max'], settings['partner_iso_low'],
                settings['partner_iso_high'], settings['purity_limit']) != ('2mu2e', *bounds, 0., .5, .1):
            raise AssertionError('Prior 2D distribution production conventions differ.')
        if (settings['current_hist'], settings['expected_current_channel']) != (
                config['current_hist'], config['expected_current_channel']):
            raise AssertionError('Prior 2D display uses another histogram or channel.')
        summary = finalize_buffer_distributions_pd.read_csv(directory/'summary.csv').set_index('candidate')
        figures = finalize_buffer_distributions_pd.read_csv(directory/'figure_manifest.csv')
        data_figure = figures.loc[figures.figure.eq('data_VR_planes')]
        if len(data_figure) != 1:
            raise AssertionError('The pre-existing 2D DATA figure must be recorded.')
        paths = [directory/'manifest.json', directory/'summary.csv', directory/'distribution_bin_purity.csv',
                 directory/'figure_manifest.csv', directory/data_figure.iloc[0].png]
        for path in paths:
            if not path.is_file():
                raise AssertionError('Missing evidence of an already displayed 2D distribution: ' + str(path))
            inputs.append(finalize_buffer_distributions_common.candidate_input_record(path, 'Already published exact folded VR DATA bins'))
        frame = finalize_buffer_distributions_pd.read_csv(directory/'distribution_bin_purity.csv', float_precision='round_trip')
        if not frame.signal_hypotheses.eq(120).all() or frame.duplicated(['plane', 'bin_x', 'bin_y']).any():
            raise AssertionError('Each prior 2D bin requires all 120 hypotheses and a unique identity.')
        for row in frame.to_dict('records'):
            state = row['plane']
            record = dict(row, scope=prefix+'_'+state, mass_slice=mass, ix=int(row['bin_x']), iy=int(row['bin_y']),
                xlow=row['x_low'], xhigh=row['x_high'], ylow=row['y_low'], yhigh=row['y_high'],
                physical_xlow=row['x_low'], physical_xhigh=finalize_buffer_distributions_np.inf if finalize_buffer_distributions_np.isclose(row['x_high'], 2.) else row['x_high'],
                physical_ylow=-finalize_buffer_distributions_np.inf if finalize_buffer_distributions_np.isclose(row['y_low'], 0.) else row['y_low'],
                physical_yhigh=finalize_buffer_distributions_np.inf if finalize_buffer_distributions_np.isclose(row['y_high'], finalize_buffer_distributions_np.pi) else row['y_high'],
                bin_approved=bool(row['approved']),
                previously_displayed=bool(summary.loc[state, 'nominal_approved'] and row['approved']))
            rows.append(record)
        validation.append(dict(scope=folder,
            check='Prior DATA figure, integrated display gate and exact 120-hypothesis folded-bin mask are present and recorded', passed=True))
    return finalize_buffer_distributions_pd.DataFrame(rows)


def finalize_buffer_distributions__physical_bins(xedges, yedges):
    rows = []
    for ix, iy in finalize_buffer_distributions_np.ndindex(len(xedges)-1, len(yedges)-1):
        rows.append(dict(ix=ix, iy=iy, xlow=xedges[ix], xhigh=xedges[ix+1], ylow=yedges[iy], yhigh=yedges[iy+1],
            physical_xlow=xedges[ix], physical_xhigh=finalize_buffer_distributions_np.inf if ix == len(xedges)-2 else xedges[ix+1],
            physical_ylow=-finalize_buffer_distributions_np.inf if iy == 0 else yedges[iy],
            physical_yhigh=finalize_buffer_distributions_np.inf if iy == len(yedges)-2 else yedges[iy+1]))
    return finalize_buffer_distributions_pd.DataFrame(rows)


def finalize_buffer_distributions__native_atomic(mc, state):
    reference = mc['backgrounds'][state+':low']
    x = finalize_buffer_distributions_np.r_[reference['xedges'], finalize_buffer_distributions_np.inf]
    y = finalize_buffer_distributions_np.r_[-finalize_buffer_distributions_np.inf, reference['yedges'], finalize_buffer_distributions_np.inf]
    atoms = finalize_buffer_distributions_pd.DataFrame([dict(ix=i, iy=j, xlow=x[i], xhigh=x[i+1], ylow=y[j], yhigh=y[j+1])
                          for i in range(len(x)-1) for j in range(len(y)-1)])
    leaves = [(0., 150., 'all', state+':low')]
    if state == 'PP':
        leaves += [(150., 200., 'all', 'fine:PP_150_200_all'), (200., 300., 'all', 'fine:PP_200_300_all')]
    elif state == 'PD':
        leaves += [(lo, hi, prompt, f'fine:PD_{lo}_{hi}_{prompt}') for lo, hi in [(150, 200), (200, 300)] for prompt in ('3', 'ge4')]
    elif state == 'DP':
        leaves += [(150., 300., 'all', 'DP:intermediate')]
    if state != 'DD':
        leaves += [(300., finalize_buffer_distributions_np.inf, 'all', state+':high')]
    names = sorted(mc['signals'][state+':low'])
    pieces, value, variance, signal, signal_variance = [], [], [], [], []
    for lo, hi, prompt, name in leaves:
        plane = mc['backgrounds'][name]
        pieces.append(atoms.assign(mass_low=lo, mass_high=hi, prompt_group=prompt, leaf=name))
        value.append(plane['values'][1:, :].ravel())
        variance.append(plane['variances'][1:, :].ravel())
        signal.append(finalize_buffer_distributions_np.asarray([mc['signals'][name][sample]['values'][1:, :].ravel() for sample in names]))
        signal_variance.append(finalize_buffer_distributions_np.asarray([mc['signals'][name][sample]['variances'][1:, :].ravel() for sample in names]))
    return dict(atoms=finalize_buffer_distributions_pd.concat(pieces, ignore_index=True), values=finalize_buffer_distributions_np.concatenate(value), variances=finalize_buffer_distributions_np.concatenate(variance),
        signals=finalize_buffer_distributions_np.concatenate(signal, axis=1), signal_variances=finalize_buffer_distributions_np.concatenate(signal_variance, axis=1), signal_names=names)


def finalize_buffer_distributions__bin_mask(atomic, cell, mass):
    atoms = atomic['atoms']
    lo, hi = {'low': (0., 150.), 'intermediate': (150., 300.), 'all': (0., finalize_buffer_distributions_np.inf)}[mass]
    return ((atoms.xlow >= cell['physical_xlow']) & (atoms.xhigh <= cell['physical_xhigh'])
            & (atoms.ylow >= cell['physical_ylow']) & (atoms.yhigh <= cell['physical_yhigh'])
            & (atoms.mass_low >= lo) & (atoms.mass_high <= hi)).to_numpy(bool)


def finalize_buffer_distributions__regress_buffer_mc(plane, current, scope):
    expected = current.loc[current.source.eq('MC') & current.scope.eq(scope) & current.candidate_id.eq(finalize_buffer_distributions_BUFFER['candidate_id'])]
    if len(expected) != 1:
        raise AssertionError('Every buffered VR plane needs one established MC reference.')
    expected = expected.iloc[0]
    for region, coordinate in finalize_buffer_distributions_candidate_coordinates(finalize_buffer_distributions_BUFFER).items():
        actual = finalize_buffer_distributions_common._comparison_regions(plane, *coordinate)[region]
        finalize_buffer_distributions_np.testing.assert_allclose([actual['value'], actual['variance']], [expected[region], expected['var_'+region]], rtol=1e-8, atol=1e-8)


def finalize_buffer_distributions__dd_mc_plane(weights, variances, edges, nominal):
    partner = slice(finalize_buffer_distributions_common._comparison_split(edges['iso1'], 0.), finalize_buffer_distributions_common._comparison_split(edges['iso1'], .5))
    muon = slice(None, finalize_buffer_distributions_common._comparison_split(edges['disp0'], 2.5))
    egm = slice(finalize_buffer_distributions_common._comparison_split(edges['disp1'], .5), None)
    mass = slice(finalize_buffer_distributions_common._comparison_split(edges['ljlj_mass'], 150.), None) if nominal else slice(
        finalize_buffer_distributions_common._comparison_split(edges['ljlj_mass'], 0.), finalize_buffer_distributions_common._comparison_split(edges['ljlj_mass'], 150.))
    reduced = [array[:, partner, muon, egm, :, mass].sum(axis=(1, 2, 3, 5)) for array in (weights, variances)]
    return dict(finalize_buffer_distributions_common._comparison_plane(*reduced, edges['iso0'], edges['abs_dphi']), kind='mu_dphi')


def finalize_buffer_distributions_load_buffer_mc(config):
    """One MC-only pass produces the prior fine planes plus low/nominal DD."""
    bkg = finalize_buffer_distributions_Path(config['bkg_dir'])
    settings = dict(config, bkg_dir=bkg, signal_dir=bkg.with_name(bkg.name.replace('_bkg_', '_signal_')))
    backgrounds = sorted(bkg.glob('*.coffea'))
    signals = finalize_buffer_distributions_common.candidate_signal_paths(settings)
    if len(backgrounds) != 18 or len(signals) != 120:
        raise AssertionError('Require the same 18 background and 120 signal MC samples.')
    totals, signal_planes, manifest, checks = {}, {}, [], []
    jobs = [(path, 'Background') for path in backgrounds] + [(path, 'Signal') for path in signals]
    for index, (path, source) in enumerate(jobs, 1):
        loaded, payload = finalize_buffer_distributions_common._comparison_payload(path, 'sixd')
        weights, variances, edges = finalize_buffer_distributions_common._comparison_current_cube(payload, path, settings)
        fine = finalize_buffer_distributions_project_strata(weights, variances, edges, finalize_buffer_distributions_strata_specs(), settings)
        planes = {'fine:'+name: plane for name, plane in fine.items()}
        for state in ('PP', 'PD', 'DP'):
            partitions, _ = finalize_buffer_distributions_project_mass_partitions(weights, variances, edges, state, settings)
            planes.update({state+':'+mass: plane for mass, plane in partitions.items()})
        planes.update({'DD:low': finalize_buffer_distributions__dd_mc_plane(weights, variances, edges, False),
                       'DD:nominal': finalize_buffer_distributions__dd_mc_plane(weights, variances, edges, True)})
        for name, plane in planes.items():
            if source == 'Background':
                totals[name] = finalize_buffer_distributions_common.add_candidate(totals.get(name), plane, finalize_buffer_distributions_common._comparison_group(path.stem))
            else:
                signal_planes.setdefault(name, {})[path.stem] = plane
        manifest.append(finalize_buffer_distributions_common.candidate_input_record(path, 'Buffered distribution native '+source+' MC'))
        checks.append(dict(scope=path.stem, check='MC-only native/fine and low/nominal DD distributions use the established six-dimensional production', passed=True))
        del loaded, payload, weights, variances, fine, planes
        finalize_buffer_distributions_gc.collect()
        if index % 20 == 0 or index == len(jobs):
            print(f'09 buffered distribution MC {index}/{len(jobs)}', flush=True)
    return dict(backgrounds=totals, signals=signal_planes, manifest=manifest, validation=checks)


def finalize_buffer_distributions__load_data_bins(config, approved):
    if set(approved.scope) != finalize_buffer_distributions_ALLOWED_DATA_SCOPES or not approved.complementary_pass.all():
        raise AssertionError('Only exact previously displayed bins in four selected-buffer VRs may be read.')
    bkg = finalize_buffer_distributions_Path(config['bkg_dir'])
    directory = bkg.with_name(bkg.name.replace('_bkg_', '_data_'))
    paths = sorted(directory.glob('*.coffea'))
    original = finalize_buffer_distributions_pd.read_csv(finalize_buffer_distributions_Path(config['vr2_output_dir'])/'input_manifest.csv')
    original = original.loc[original.source.eq('DATA selected VR candidates')].set_index('path')
    if len(paths) != 68 or set(map(str, paths)) != set(original.index):
        raise AssertionError('Require the established 68 original DATA files.')
    manifest = []
    for path in paths:
        record = finalize_buffer_distributions_common.candidate_input_record(path, 'Exact prior VR display bins only; no nominal or full-plane DATA query')
        expected = original.loc[str(path)]
        if (record['sha256'], record['size_bytes']) != (expected.sha256, expected.size_bytes):
            raise AssertionError('An original DATA file differs from the established 01/03 production.')
        manifest.append(record)
    records = approved.to_dict('records')
    totals = {(row['scope'], row['ix'], row['iy']): 0. for row in records}
    for index, path in enumerate(paths, 1):
        loaded, payload = finalize_buffer_distributions_common._extension_data_payload(path)
        weights, variances, edges = finalize_buffer_distributions_common._comparison_current_cube(payload, path, config)
        for row in records:
            if row['plane'] not in ('PP', 'PD') or row['mass_slice'] not in ('low', 'intermediate'):
                raise AssertionError('The DATA-bin whitelist excludes nominal DD and mass-inclusive queries.')
            selection = list(finalize_buffer_distributions__state_index(edges, row['plane'], config))
            lo, hi = (0., 150.) if row['mass_slice'] == 'low' else (150., 300.)
            selection.append(slice(finalize_buffer_distributions_common._comparison_split(edges['ljlj_mass'], lo), finalize_buffer_distributions_common._comparison_split(edges['ljlj_mass'], hi)))
            selection[0] = slice(finalize_buffer_distributions_common._comparison_split(edges['iso0'], row['physical_xlow']),
                None if not finalize_buffer_distributions_np.isfinite(row['physical_xhigh']) else finalize_buffer_distributions_common._comparison_split(edges['iso0'], row['physical_xhigh']))
            selection[4] = slice(None if not finalize_buffer_distributions_np.isfinite(row['physical_ylow']) else finalize_buffer_distributions_common._comparison_split(edges['abs_dphi'], row['physical_ylow']),
                None if not finalize_buffer_distributions_np.isfinite(row['physical_yhigh']) else finalize_buffer_distributions_common._comparison_split(edges['abs_dphi'], row['physical_yhigh']))
            values, vs = weights[tuple(selection)], variances[tuple(selection)]
            if (not finalize_buffer_distributions_np.isfinite(values).all() or (values < 0).any()
                    or not finalize_buffer_distributions_np.array_equal(values, vs) or not finalize_buffer_distributions_np.array_equal(values, finalize_buffer_distributions_np.rint(values))):
                raise AssertionError('Every approved DATA slice must have integer nonnegative counts and sumw2=N.')
            totals[(row['scope'], row['ix'], row['iy'])] += float(values.sum())
        del loaded, payload, weights, variances
        finalize_buffer_distributions_gc.collect()
        if index % 20 == 0 or index == len(paths):
            print(f'09 pre-existing VR DATA display bins {index}/{len(paths)}', flush=True)
    return totals, manifest


def finalize_buffer_distributions__validate_visible_counts(bins, current, validation):
    visible = bins.loc[bins.display_allowed]
    if not finalize_buffer_distributions_np.array_equal(visible.data, visible.var_data) or not finalize_buffer_distributions_np.array_equal(visible.data, finalize_buffer_distributions_np.rint(visible.data)):
        raise AssertionError('Displayed DATA values must remain integer counts with equal variances.')
    for scope in finalize_buffer_distributions_ALLOWED_DATA_SCOPES:
        frame = visible.loc[visible.scope.eq(scope)]
        reference = current.loc[current.source.eq('DATA') & current.scope.eq(scope)
                                & current.candidate_id.eq(finalize_buffer_distributions_BUFFER['candidate_id'])].iloc[0]
        for region in 'ABCD':
            high_x = frame.physical_xlow >= finalize_buffer_distributions_BUFFER['b_iso']
            low_x = frame.physical_xhigh <= finalize_buffer_distributions_BUFFER['a_iso']
            high_y = frame.physical_ylow >= finalize_buffer_distributions_BUFFER['a_phi']
            low_y = frame.physical_yhigh <= finalize_buffer_distributions_BUFFER['c_phi']
            choose = (low_x if region in 'AC' else high_x) & (high_y if region in 'AB' else low_y)
            if frame.loc[choose, 'data'].sum() > reference[region]:
                raise AssertionError('Visible bin counts exceed the established integrated DATA region.')
        validation.append(dict(scope=scope, check='Every displayed DATA bin has variance=N and each visible ABCD subset is bounded by the existing regional count', passed=True))


def finalize_buffer_distributions__regress_nominal_mc(plane, nominal):
    expected = nominal.loc[nominal.candidate_id.eq(finalize_buffer_distributions_BUFFER['candidate_id'])]
    if len(expected) != 1 or not expected.source.eq('MC').all():
        raise AssertionError('Nominal MC regression requires the established MC-only buffer row.')
    expected = expected.iloc[0]
    for region, coordinate in finalize_buffer_distributions_candidate_coordinates(finalize_buffer_distributions_BUFFER).items():
        actual = finalize_buffer_distributions_common._comparison_regions(plane, *coordinate)[region]
        finalize_buffer_distributions_np.testing.assert_allclose([actual['value'], actual['variance']],
            [expected[region], expected['var_'+region]], rtol=1e-8, atol=1e-8)


# ========================================================================
# abcd_inclusive_dp_bins: original study API
# ========================================================================
"""Prospective MC-only disclosure audit for mass-inclusive DP display bins.

The exact prior observation set includes every public 01/03/07 DP grid row,
their existing display bins, both earlier 09 reports and the frozen 261-row
scan release.  Proposed bins are accepted in fixed native-index order only
after the full native rectangle/mass-subset guard and the established joint
complement audit pass.  This module never opens an original DATA file.
"""
from pathlib import Path as finalize_inclusive_dp_bins_Path
from time import monotonic as finalize_inclusive_dp_bins_monotonic

import numpy as finalize_inclusive_dp_bins_np
import pandas as finalize_inclusive_dp_bins_pd

finalize_inclusive_dp_bins_buffered = _FINALIZE_NAMESPACES['abcd_buffer_distributions']
finalize_inclusive_dp_bins_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_inclusive_dp_bins_scan_validation = _FINALIZE_NAMESPACES['abcd_sideband_scan_validation']


def finalize_inclusive_dp_bins_audit_inclusive_dp_bins(config, historical, previous, latest, mc, prior_bins=None):
    """Return MC-only audit tables, metadata and the approved physical-bin rows.

    ``prior_bins`` may contain the combined 01/03/07 display-prior records from
    ``abcd_selected_vr_displays``.  Supplied records are checked against their
    source files.  The accepted set is safe in the recorded fixed order; it is
    not asserted to be the largest possible accepted subset.
    """
    started=finalize_inclusive_dp_bins_monotonic()
    if (config['topology'],config['purity_limit'],config['partner_iso_low'],config['partner_iso_high']) != ('2mu2e',.1,0.,.5):
        raise ValueError('Require the established 2mu2e, partner [0,.5), strict 10% selection.')
    historical=historical.get('tables',historical)
    previous=previous.get('tables',previous)
    latest=latest.get('tables',latest)
    inputs,validation=[],[]
    finalize_inclusive_dp_bins_scan_validation._audit_prior_snapshots(config,historical,previous,inputs,validation)
    finalize_inclusive_dp_bins_buffered._audit_latest_snapshot(config,latest,inputs,validation)
    finalize_inclusive_dp_bins__validate_original_mc(config,mc,inputs,validation)
    prior=finalize_inclusive_dp_bins__validated_display_priors(config,prior_bins,inputs,validation)
    atomic=finalize_inclusive_dp_bins_buffered._native_atomic(mc,'DP')
    if len(atomic['values'])!=624 or len(atomic['signal_names'])!=120:
        raise AssertionError('Require 624 native DP atoms and the same 120 signal hypotheses.')
    for field in ('values','variances','signals','signal_variances'):
        if not finalize_inclusive_dp_bins_np.isfinite(atomic[field]).all() or (atomic[field]<0).any():
            raise AssertionError('Every native MC atom must have finite nonnegative yields and variances.')
    finalize_inclusive_dp_bins__validate_display_mc(atomic,prior,validation)
    published=finalize_inclusive_dp_bins_scan_validation._prior_masks(historical,previous,'DP',atomic)
    provenance=[dict(source='Historical 09 mass/strata/sideband releases',public_query_masks=len(published))]
    finalize_inclusive_dp_bins__add_public_integrals(published,atomic,latest['sv_closure'],'frozen261',provenance)
    for folder,mass,expected in [('01_2mu2e_VR1_summary_outputs','low',85),
                                  ('03_2mu2e_VR2_summary_outputs','intermediate',72),
                                  ('07_2mu2e_comprehensive_outputs','all',4)]:
        path=finalize_inclusive_dp_bins_Path(config['notebook_path']).parent/folder/('mass_grid.csv' if mass=='all' else 'grid.csv')
        inputs.append(finalize_inclusive_dp_bins_common.candidate_input_record(path,'Complete previously public DP grid measurements'))
        frame=finalize_inclusive_dp_bins_pd.read_csv(path,float_precision='round_trip')
        finalize_inclusive_dp_bins__validate_grid_mc(frame,mc['backgrounds']['DP:'+mass],mass,validation)
        count=finalize_inclusive_dp_bins__add_public_integrals(published,atomic,frame,folder,provenance,mass)
        if count!=expected:
            raise AssertionError('The established full DP grid release changed: '+folder)
    oldbins=prior.loc[prior.previously_displayed]
    if oldbins.groupby('mass_slice').size().to_dict()!={'low':42,'intermediate':44}:
        raise AssertionError('Require the 86 previously displayed low/intermediate DP bins and no inclusive DP bins.')
    for row in oldbins.to_dict('records'):
        published[f'oldbin:{row["scope"]}:{row["ix"]}:{row["iy"]}']=finalize_inclusive_dp_bins_buffered._bin_mask(atomic,row,row['mass_slice'])
    published=finalize_inclusive_dp_bins_scan_validation._unique_masks(published)
    initial_masks=len(published)
    if initial_masks!=728:
        raise AssertionError('The complete established DP prior must contain 728 unique physical query masks.')
    atoms=atomic['atoms']
    xedges=sorted(set(atoms.xlow)|set(atoms.xhigh))
    yedges=sorted(set(atoms.ylow)|set(atoms.yhigh))
    guard=finalize_inclusive_dp_bins_scan_validation._exhaustive_guard(atomic,published,xedges,yedges)
    if (guard['total_rectangles'],guard['grandfathered'])!=(86632,767):
        raise AssertionError('The exact native rectangle/mass-subset prior audit changed.')
    eligible=prior.loc[prior.scope.eq('INCLUSIVE_DP')&prior.bin_approved].sort_values(['ix','iy'])
    if len(eligible)!=77:
        raise AssertionError('Require exactly 77 central-MC-pure inclusive DP display bins.')
    rows,details=[],[]
    for position,record in enumerate(eligible.to_dict('records'),1):
        name=f'newDPinclusive:{record["ix"]}:{record["iy"]}'
        mask=finalize_inclusive_dp_bins_buffered._bin_mask(atomic,record,'all')
        b=float(atomic['values'][mask].sum())
        signal=atomic['signals'][:,mask].sum(axis=1)
        maximum=float((signal/b).max())
        if not b>1e-8 or not maximum<.1:
            raise AssertionError('A proposed DP bin must retain positive MC support and all-120 strict purity.')
        probe=finalize_inclusive_dp_bins_scan_validation._guard_probe(guard,{name:mask})
        exhaustive=probe['audit']
        joint=(finalize_inclusive_dp_bins_scan_validation.joint_disclosure_audit(published,{name:mask},atomic)
               if exhaustive.empty else finalize_inclusive_dp_bins_pd.DataFrame())
        unsafe_joint=joint.loc[~joint.approved] if len(joint) else joint
        accepted=bool(exhaustive.empty and unsafe_joint.empty)
        rank_increment=len(probe['increment'])
        if accepted:
            finalize_inclusive_dp_bins_scan_validation._guard_commit(guard,probe)
            published[name]=mask
        for kind,frame in [('exhaustive',exhaustive),('joint',joint)]:
            if len(frame):
                details.extend(frame.assign(candidate_bin=name,stage=kind).to_dict('records'))
        rows.append(dict(scope='INCLUSIVE_DP',plane='DP',mass_slice='all',ix=record['ix'],iy=record['iy'],
            physical_xlow=record['physical_xlow'],physical_xhigh=record['physical_xhigh'],
            physical_ylow=record['physical_ylow'],physical_yhigh=record['physical_yhigh'],
            background=b,var_background=float(atomic['variances'][mask].sum()),max_S_B=maximum,
            signal_hypotheses=120,bin_purity_pass=True,previously_displayed=False,
            already_identifiable=(rank_increment==0),accepted=accepted,
            rank_increment=rank_increment if accepted else 0,
            unsafe_exhaustive=len(exhaustive),unsafe_joint=len(unsafe_joint),
            reason='PASS_NATIVE_RECTANGLE_MASS_AND_JOINT_COMPLEMENT_AUDITS' if accepted else
                   'UNSAFE_NATIVE_RECTANGLE_MASS_COMPLEMENT' if len(exhaustive) else 'UNSAFE_JOINT_COMPLEMENT'))
        if position%20==0 or position==len(eligible):
            print(f'Inclusive DP prospective bin audit {position}/{len(eligible)}; accepted {sum(row["accepted"] for row in rows)}',flush=True)
    newly_unsafe=finalize_inclusive_dp_bins_scan_validation._validate_final_exhaustive_basis(guard,published)
    audit=finalize_inclusive_dp_bins_pd.DataFrame(rows)
    accepted=audit.loc[audit.accepted].copy()
    if (len(accepted),int(accepted.rank_increment.sum()),newly_unsafe)!=(21,20,0):
        raise AssertionError('Require the reproducible 21-bin accepted set, 20 new functionals and zero newly identifiable unsafe native targets.')
    validation.extend([
        dict(scope='DP prior observations',check='728 distinct existing masks include all historical, frozen scan, complete 01/03/07 public grids and 86 old display bins',passed=True),
        dict(scope='DP prospective display',check='77 all-120-pure bins are considered in fixed ix/iy order; 21 jointly pass exhaustive and partition/nested/remainder checks',passed=True),
        dict(scope='DP independent final QR',check='Independent full-row-space QR returns zero newly identifiable unsafe native rectangle/mass-subset targets',passed=newly_unsafe==0),
        dict(scope='DP DATA access',check='This audit uses only MC and already published records; no original DATA file is opened',passed=True),
    ])
    metadata=dict(original_data_files_opened=0,signal_hypotheses=120,prior_display_bins=len(oldbins),
        prior_masks=initial_masks,final_masks=len(published),candidate_pure_bins=len(eligible),
        accepted_bins=len(accepted),new_linear_functionals=int(accepted.rank_increment.sum()),
        native_rectangle_mass_subset_targets=guard['total_rectangles'],
        grandfathered_unsafe_targets=guard['grandfathered'],
        newly_identifiable_unsafe_targets=newly_unsafe,independent_final_QR_pass=newly_unsafe==0,
        ordering='Native displayed ix then iy, without any DATA-dependent ordering; commit only after both audit families pass.',
        proof_scope='Every native iso/angle rectangle times every nonempty DP low/intermediate/high mass-leaf union, plus the established joint partition/nested-complement/remainder criterion. Not a claim about every arbitrary Boolean subset.',
        audit_runtime_seconds=finalize_inclusive_dp_bins_monotonic()-started)
    return dict(tables=dict(dp_bin_audit=audit,dp_complement_details=finalize_inclusive_dp_bins_pd.DataFrame(details),
        dp_prior_sources=finalize_inclusive_dp_bins_pd.DataFrame(provenance),dp_input_manifest=finalize_inclusive_dp_bins_pd.DataFrame(inputs).drop_duplicates(['path','sha256']),
        dp_validation=finalize_inclusive_dp_bins_pd.DataFrame(validation)),metadata=metadata,accepted_bins=accepted)


def finalize_inclusive_dp_bins__validate_original_mc(config,mc,inputs,validation):
    path=finalize_inclusive_dp_bins_Path(config['vr2_output_dir'])/'input_manifest.csv'
    inputs.append(finalize_inclusive_dp_bins_common.candidate_input_record(path,'Established 18 background and 120 signal MC source identities'))
    original=finalize_inclusive_dp_bins_pd.read_csv(path)
    original=original.loc[original.source.isin(['Background','Signal hypothesis'])].set_index('path')
    current=finalize_inclusive_dp_bins_pd.DataFrame(mc['manifest']).set_index('path')
    if len(current)!=138 or len(original)!=138 or set(current.index)!=set(original.index):
        raise AssertionError('Require exactly the established 18+120 MC files.')
    for filename,row in current.iterrows():
        expected=original.loc[filename]
        actual=finalize_inclusive_dp_bins_common.candidate_input_record(finalize_inclusive_dp_bins_Path(filename),'DP prospective audit original MC identity')
        if (row.sha256,row.size_bytes)!=(expected.sha256,expected.size_bytes):
            raise AssertionError('The supplied MC projection source identity changed: '+filename)
        if (actual['sha256'],actual['size_bytes'])!=(expected.sha256,expected.size_bytes):
            raise AssertionError('An original MC file changed after its recorded projection: '+filename)
        inputs.append(actual)
    validation.append(dict(scope='DP MC provenance',check='Every current and supplied MC path, size and SHA256 matches the original 18+120 production',passed=True))


def finalize_inclusive_dp_bins__validated_display_priors(config,supplied,inputs,validation):
    # Local import permits the display module to call this audit with its own
    # prior records without creating a module-import cycle.
    _inclusive_display_priors = _FINALIZE_NAMESPACES['abcd_selected_vr_displays']._inclusive_display_priors
    canonical=finalize_inclusive_dp_bins_pd.concat([finalize_inclusive_dp_bins_buffered._display_priors(config,inputs,validation),
                         _inclusive_display_priors(config,inputs,validation)],ignore_index=True)
    canonical=canonical.loc[canonical.plane.eq('DP')].sort_values(['scope','ix','iy']).reset_index(drop=True)
    fields=['scope','plane','mass_slice','ix','iy','physical_xlow','physical_xhigh',
            'physical_ylow','physical_yhigh','background','background_variance','max_S_B',
            'bin_approved','previously_displayed']
    if len(canonical)!=462 or canonical.duplicated(['scope','ix','iy']).any():
        raise AssertionError('Require 154 unique display bins for each DP mass selection.')
    if supplied is not None:
        check=supplied.loc[supplied.plane.eq('DP')].sort_values(['scope','ix','iy']).reset_index(drop=True)
        finalize_inclusive_dp_bins_pd.testing.assert_frame_equal(check[fields],canonical[fields],check_dtype=False)
    return canonical


def finalize_inclusive_dp_bins__validate_display_mc(atomic,prior,validation):
    for row in prior.to_dict('records'):
        mask=finalize_inclusive_dp_bins_buffered._bin_mask(atomic,row,row['mass_slice'])
        b=float(atomic['values'][mask].sum())
        vb=float(atomic['variances'][mask].sum())
        signal=atomic['signals'][:,mask].sum(axis=1)
        maximum=float((signal/b).max()) if b>1e-8 else finalize_inclusive_dp_bins_np.nan
        finalize_inclusive_dp_bins_np.testing.assert_allclose([b,vb,maximum],[row['background'],row['background_variance'],row['max_S_B']],
                                   rtol=1e-8,atol=1e-8,equal_nan=True)
        if bool(b>1e-8 and finalize_inclusive_dp_bins_np.isfinite(maximum) and maximum<.1)!=bool(row['bin_approved']):
            raise AssertionError('A previous exact folded DP display-bin purity decision changed.')
    validation.append(dict(scope='DP native MC',check='All 462 exact folded DP bins reproduce prior background, variance and all-120 contamination envelopes',passed=True))


def finalize_inclusive_dp_bins__add_public_integrals(published,atomic,frame,label,provenance,mass=None):
    chosen=frame.loc[frame.source.eq('DATA')&frame.plane.eq('DP')]
    gate='display_allowed' if 'display_allowed' in chosen else 'approved'
    chosen=chosen.loc[chosen[gate]]
    for index,row in enumerate(chosen.to_dict('records')):
        spec=row if 'a_iso' in row else dict(a_iso=row['tx'],b_iso=row['tx'],a_phi=row['ty'],c_phi=row['ty'])
        scope=mass if mass is not None else row['mass_slice']
        for region,mask in finalize_inclusive_dp_bins_scan_validation._scope_masks(spec,atomic,scope).items():
            published[f'{label}:{index}:{region}']=mask
    provenance.append(dict(source=label,public_integral_definitions=len(chosen)))
    return len(chosen)


def finalize_inclusive_dp_bins__validate_grid_mc(frame,plane,mass,validation):
    rows=frame.loc[frame.source.eq('MC')&frame.plane.eq('DP')]
    if len(rows)!=154 or rows.duplicated(['tx','ty']).any():
        raise AssertionError('Every prior DP MC grid must retain its 154 native coordinates.')
    for row in rows.itertuples():
        actual=finalize_inclusive_dp_bins_common._comparison_regions(plane,row.tx,row.ty)
        for region in 'ABCD':
            finalize_inclusive_dp_bins_np.testing.assert_allclose([actual[region]['value'],actual[region]['variance']],
                [getattr(row,region),getattr(row,'var_'+region)],rtol=1e-8,atol=1e-8)
    validation.append(dict(scope='DP '+mass,check='All 154 original DP background-MC grid coordinates reproduce their source sumw and sumw2',passed=True))


# ========================================================================
# abcd_selected_vr_displays: original study API
# ========================================================================
"""Selected loose/buffer VR displays with separate bin and closure safeguards.

The 2D display gate is an exact-bin redisplay gate, independent of the selected
WP's integrated closure gate.  An unsafe ABCD integral does not imply every
constituent bin is unsafe.  A 2D bin is shown only if its identical physical
query was previously displayed in 01/03/07 and still passes all-120 purity,
or a prospective joint-disclosure audit explicitly accepts the inclusive DP bin.
No new integrated ABCD DATA measurement is introduced by either overlay.
"""
from pathlib import Path as finalize_selected_vr_displays_Path
import copy as finalize_selected_vr_displays_copy
import gc as finalize_selected_vr_displays_gc
import json as finalize_selected_vr_displays_json

import numpy as finalize_selected_vr_displays_np
import pandas as finalize_selected_vr_displays_pd

finalize_selected_vr_displays_buffered = _FINALIZE_NAMESPACES['abcd_buffer_distributions']
finalize_selected_vr_displays_common = _FINALIZE_NAMESPACES['abcd_finalize_common']
finalize_selected_vr_displays_build_iso_focus = _FINALIZE_NAMESPACES['abcd_iso_focus'].build_iso_focus
finalize_selected_vr_displays_candidate_coordinates = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].candidate_coordinates


finalize_selected_vr_displays_SELECTIONS = {
    'loose': dict(candidate_id='L_i0p25_p2p6', a_iso=.25, a_phi=2.6, b_iso=.25,
                  c_phi=2.6, display_label='Selected loose: isolation < 0.25, |dphi| >= 2.6'),
    'gap': dict(candidate_id='G_i0p25_p2', a_iso=.2, a_phi=2.7, b_iso=.25,
                c_phi=2., display_label='Selected buffer: A=(0.20,2.7); sidebands=(0.25,2.0)'),
}


def finalize_selected_vr_displays_run_selected_vr_displays(config, historical, previous_sideband, latest_scan,
                            iso_result=None, load_data=False, mc=None, final_gap_result=None):
    """Audit previous bins and proposed DP bins before any original DATA query."""
    historical = historical.get('tables', historical)
    previous_sideband = previous_sideband.get('tables', previous_sideband)
    latest_scan = latest_scan.get('tables', latest_scan)
    supplied_mc = mc is not None
    mc = finalize_selected_vr_displays_buffered.load_buffer_mc(config) if mc is None else mc
    base = finalize_selected_vr_displays_buffered.run_buffer_distributions(config, historical, previous_sideband,
                                             latest_scan, load_data=False, mc=mc)
    inputs = base['tables']['bd_input_manifest'].to_dict('records')
    validation = base['tables']['bd_validation'].to_dict('records')
    for check in validation:
        if check['scope'] in ('Prior information','Joint protection','Blinding'):
            check['check'] = 'Original 502-bin buffer redisplay audit only: '+check['check']
    prior = finalize_selected_vr_displays_buffered._display_priors(config, inputs, validation)
    prior = finalize_selected_vr_displays_pd.concat([prior, finalize_selected_vr_displays__inclusive_display_priors(config, inputs, validation)], ignore_index=True)
    bins = base['tables']['bd_bins'].copy()
    bins['previously_displayed'] = False
    bins['complementary_pass'] = False
    bins['release_eligible'] = False
    bins['display_allowed'] = False
    bins['data'] = finalize_selected_vr_displays_np.nan
    bins['var_data'] = finalize_selected_vr_displays_np.nan
    bins['release_provenance'] = 'WITHHELD'
    bins['joint_release_pass'] = False
    bins['already_identifiable'] = False
    bins['rank_increment'] = 0
    bins = bins.drop(columns=['integrated_allowed'])
    atoms = {state: finalize_selected_vr_displays_buffered._native_atomic(mc, state) for state in ('DD', 'PP', 'PD', 'DP')}
    proof = []
    for index, row in bins.iterrows():
        old = prior.loc[prior.scope.eq(row.scope) & prior.ix.eq(row.ix) & prior.iy.eq(row.iy)]
        if row.mass_slice == 'nominal':
            bins.loc[index, 'reason'] = 'NOMINAL_DATA_ALWAYS_BLINDED'
            continue
        if len(old) != 1:
            raise AssertionError('Every nonnominal display bin requires one previous 01/03/07 record.')
        old = old.iloc[0]
        finalize_selected_vr_displays_np.testing.assert_allclose([row.background, row.var_background, row.max_S_B],
            [old.background, old.background_variance, old.max_S_B], rtol=1e-8, atol=1e-8, equal_nan=True)
        if bool(row.bin_purity_pass) != bool(old.bin_approved):
            raise AssertionError('The native MC reconstruction changed a previous bin-purity decision.')
        was_public = bool(old.previously_displayed)
        bins.loc[index, 'previously_displayed'] = was_public
        same = False
        if was_public and row.bin_purity_pass:
            proposed = finalize_selected_vr_displays_buffered._bin_mask(atoms[row.plane], row, row.mass_slice)
            previous = finalize_selected_vr_displays_buffered._bin_mask(atoms[row.plane], old, row.mass_slice)
            if not finalize_selected_vr_displays_np.array_equal(proposed, previous):
                raise AssertionError('Redisplayed DATA must equal the previous physical bin mask exactly.')
            finalize_selected_vr_displays_np.testing.assert_allclose(atoms[row.plane]['values'][proposed].sum(), row.background,
                                       rtol=1e-10, atol=1e-8)
            same = True
            proof.append(dict(scope=row.scope, plane=row.plane, mass_slice=row.mass_slice,
                ix=row.ix, iy=row.iy, exact_previous_mask=True,
                source_notebook='07' if row.mass_slice == 'all' else '01' if row.mass_slice == 'low' else '03',
                newly_released_linear_functionals=0, newly_identifiable_unsafe_regions=0,
                reason='Exact public query reused; every prior joint observation span remains unchanged.'))
        eligible = bool(was_public and row.bin_purity_pass and same)
        bins.loc[index, ['complementary_pass', 'release_eligible', 'display_allowed']] = [same, eligible, eligible and load_data]
        if eligible:
            bins.loc[index, ['release_provenance','already_identifiable']] = ['PREVIOUS_PUBLIC_BIN',True]
        bins.loc[index, 'reason'] = ('EXACT_PREVIOUS_PUBLIC_BIN_AND_ALL120_PURITY_PASS' if eligible else
            'NO_PREVIOUSLY_DISPLAYED_IDENTICAL_BIN' if not was_public else 'BIN_CONTAMINATION_OR_UNSUPPORTED_MC')
    eligible = bins.loc[bins.release_eligible]
    expected = {'VR1_DD':45, 'VR1_PP':119, 'VR1_PD':118, 'VR1_DP':42,
                'VR2_PP':128, 'VR2_PD':137, 'VR2_DP':44, 'INCLUSIVE_PP':135, 'INCLUSIVE_PD':141}
    if eligible.groupby('scope').size().to_dict() != expected:
        raise AssertionError('The exact previously displayed 01/03/07 safe-bin set changed.')
    # A new DP query is permitted only after the full joint set has passed the
    # native-rectangle/mass-union and partition/complement audits.  This audit
    # contains no DATA read, and runs before even the DATA-source hash pass.
    audit_inclusive_dp_bins = _FINALIZE_NAMESPACES['abcd_inclusive_dp_bins'].audit_inclusive_dp_bins
    dp_audit = audit_inclusive_dp_bins(config, historical, previous_sideband, latest_scan, mc, prior_bins=prior)
    finalize_selected_vr_displays__apply_dp_audit(bins, dp_audit, proof, load_data)
    inputs.extend(dp_audit['tables']['dp_input_manifest'].to_dict('records'))
    validation.extend(dp_audit['tables']['dp_validation'].to_dict('records'))
    eligible = bins.loc[bins.release_eligible]
    finalize_selected_vr_displays__validate_data_production(config, inputs, validation)
    if load_data:
        counts, manifest = finalize_selected_vr_displays__read_previous_bins(config, eligible, dp_audit)
        inputs.extend(manifest)
        for index, row in bins.loc[bins.display_allowed].iterrows():
            bins.loc[index, ['data', 'var_data']] = counts[(row.scope, int(row.ix), int(row.iy))]
    if not bins.loc[~bins.display_allowed, ['data', 'var_data']].isna().all().all():
        raise AssertionError('Hidden DATA must stay NaN, including nominal and unapproved inclusive DP bins.')
    if iso_result is None:
        iso_result = finalize_selected_vr_displays_build_iso_focus(config, latest_scan)
    focused = iso_result.get('tables', iso_result)
    inputs.extend(focused['if_input_manifest'].to_dict('records'))
    final_gap_result = (finalize_selected_vr_displays_prepare_final_gap_vr(config, latest_scan)
                        if final_gap_result is None else final_gap_result)
    gap_tables = dict(latest_scan, **final_gap_result['tables'])
    inputs.extend(gap_tables['fgvr_input_manifest'].to_dict('records'))
    validation.extend(gap_tables['fgvr_validation'].to_dict('records'))
    selected_planes, scope_audit, regional = {}, [], []
    for study, spec in finalize_selected_vr_displays_SELECTIONS.items():
        selected_planes[study] = {}
        for scope, mass, state in finalize_selected_vr_displays_buffered.SCOPES:
            part = bins.loc[bins.scope.eq(scope)].sort_values(['ix', 'iy'])
            plane = finalize_selected_vr_displays_copy.deepcopy(base['planes'][scope])
            shape = plane['MC'].shape
            plane.update(DATA=part.data.to_numpy(float).reshape(shape),
                variance_DATA=part.var_data.to_numpy(float).reshape(shape),
                approved=part.release_eligible.to_numpy(bool).reshape(shape),
                display_allowed=part.display_allowed.to_numpy(bool).reshape(shape),
                reason=part.reason.to_numpy(object).reshape(shape), geometry=spec.copy(),
                release_provenance=part.release_provenance.to_numpy(object).reshape(shape),
                joint_release_pass=part.joint_release_pass.to_numpy(bool).reshape(shape),
                complementary_pass=part.complementary_pass.to_numpy(bool).reshape(shape),
                already_identifiable=part.already_identifiable.to_numpy(bool).reshape(shape),
                rank_increment=part.rank_increment.to_numpy(int).reshape(shape),
                display_gate='All-120 bin purity AND either exact previous 01/03/07 display or prospective inclusive-DP joint-release approval; independent of the integrated closure gate.')
            gate,gate_reason = finalize_selected_vr_displays__selected_closure_gate(study,spec,scope,mass,focused,gap_tables)
            plane.update(integrated_allowed=gate, integrated_reason=gate_reason,
                         closure_gate=gate, closure_gate_reason=gate_reason)
            selected_planes[study][scope] = plane
            key = state+':'+mass
            bkg = mc['backgrounds'][key]
            names = sorted(mc['signals'][key])
            region_gates = []
            for region, coordinate in finalize_selected_vr_displays_candidate_coordinates(spec).items():
                bg = finalize_selected_vr_displays_common._comparison_regions(bkg, *coordinate)[region]
                values = finalize_selected_vr_displays_np.asarray([finalize_selected_vr_displays_common._comparison_regions(mc['signals'][key][name], *coordinate)[region]['value'] for name in names])
                valid = bool(bg['value'] > 1e-8 and finalize_selected_vr_displays_np.isfinite(values).all() and (values >= 0).all())
                worst = int(finalize_selected_vr_displays_np.argmax(values))
                maximum = float(values[worst]/bg['value']) if valid else finalize_selected_vr_displays_np.nan
                passed = bool(valid and maximum < .1)
                region_gates.append(passed)
                regional.append(dict(study=study, **spec, scope=scope, plane=state, mass_slice=mass,
                    region=region, background=bg['value'], var_background=bg['variance'],
                    max_S_B=maximum, worst_sample=names[worst], signal_hypotheses=len(names),
                    purity_pass=passed, closure_gate=gate, closure_gate_reason=gate_reason))
            if gate and not all(region_gates):
                raise AssertionError('A displayed selected VR closure requires current all-ABCD/all-120 purity.')
            scope_audit.append(dict(study=study, **spec, scope=scope, plane=state, mass_slice=mass,
                closure_gate=gate, closure_gate_reason=gate_reason,
                passing_ABCD_regions=sum(region_gates), display_bins=len(part),
                bin_purity_pass=int(part.bin_purity_pass.sum()),
                previously_public_bins=int(part.previously_displayed.sum()),
                newly_audited_bins=int(part.joint_release_pass.sum()),
                eligible_bins=int(part.release_eligible.sum()), displayed_bins=int(part.display_allowed.sum()),
                all_bins_contaminated=bool(not part.bin_purity_pass.any()),
                nominal_A_evaluated=False))
    scans = finalize_selected_vr_displays__selected_scans(focused, gap_tables)
    validation.extend([
        dict(scope='01/03/07 bins', check='909 eligible physical bins exactly equal previous public queries; no new linear DATA information is introduced', passed=True),
        dict(scope='Display and closure', check='Bin-level redisplay eligibility and selected-WP integrated closure permission remain separate', passed=True),
        dict(scope='DP new bins', check='Exactly 21 prospective joint-audited DP bins are added, with 20 independent observations and zero newly identifiable unsafe audited targets', passed=True),
        dict(scope='Blinding', check='No nominal DATA or high-mass DD query; every unapproved bin remains NaN', passed=True),
        dict(scope='1D scans', check='Every DATA A/B/C/D count is an exact previously public regional measurement; no regional total or shape is inferred from partial displayed bins', passed=True),
    ])
    if load_data:
        visible = bins.loc[bins.display_allowed]
        if not finalize_selected_vr_displays_np.array_equal(visible.data, visible.var_data) or not finalize_selected_vr_displays_np.array_equal(visible.data, finalize_selected_vr_displays_np.rint(visible.data)):
            raise AssertionError('Each displayed DATA value must be an integer with sumw2=N.')
        validation.append(dict(scope='DATA display', check='Every approved queried bin has nonnegative integer counts and variance=N', passed=True))
    return dict(selected_planes=selected_planes, selections=finalize_selected_vr_displays_copy.deepcopy(finalize_selected_vr_displays_SELECTIONS),
        tables=dict(vd_bins=bins, vd_scope_audit=finalize_selected_vr_displays_pd.DataFrame(scope_audit),
            vd_region_purity=finalize_selected_vr_displays_pd.DataFrame(regional), vd_closure1d=scans,
            vd_release_audit=finalize_selected_vr_displays_pd.DataFrame(proof), vd_validation=finalize_selected_vr_displays_pd.DataFrame(validation),
            **dp_audit['tables'],
            **final_gap_result['tables'],
            vd_input_manifest=finalize_selected_vr_displays_pd.DataFrame(inputs).drop_duplicates(['path','sha256'])),
        metadata=dict(nominal_A_evaluated=False, nominal_BCD_evaluated=False,
            original_data_files_opened=68 if load_data else 0,
            original_mc_files_opened=0 if supplied_mc else 138,
            previously_public_display_bins=909, newly_queried_display_bins=21,
            newly_released_linear_functionals=20, newly_identifiable_unsafe_regions=0,
            eligible_unique_bins=len(eligible), displayed_unique_bins=len(eligible) if load_data else 0,
            signal_hypotheses=120, flow_note=finalize_selected_vr_displays_buffered.FLOW_NOTE,
            dp_audit_metadata=finalize_selected_vr_displays_copy.deepcopy(dp_audit['metadata']),
            final_gap_vr=finalize_selected_vr_displays_copy.deepcopy(final_gap_result['metadata']),
            proof_scope=dp_audit['metadata']['proof_scope'],
            display_policy='Current bin S/B_MC<10% for every hypothesis AND either exact previous public bin or prospective joint/native-rectangle/mass-union approval for an inclusive DP bin. An integrated closure failure does not imply every bin is contaminated.',
            closure_policy='Ordinary loose and buffered DATA closure require current all-region purity and exact previously public 01/03/07 or 09 regional measurements. Buffer A/B/C/D may each reuse a different approved native-grid coordinate with identical physical region and mass/displacement selection. No new integrated ABCD DATA query is made, and partial 2D displays are not used to calculate closure.'))


def finalize_selected_vr_displays__inclusive_display_priors(config, inputs, validation):
    directory = finalize_selected_vr_displays_Path(config['comprehensive_dir'])
    manifest = finalize_selected_vr_displays_json.loads((directory/'manifest.json').read_text())
    settings = manifest['configuration']
    for key in ('topology', 'current_hist', 'expected_current_channel', 'partner_iso_low', 'partner_iso_high', 'purity_limit'):
        if settings[key] != config[key]:
            raise AssertionError('The previous inclusive 2D population differs: '+key)
    closure = finalize_selected_vr_displays_pd.read_csv(directory/'mass_closure.csv')
    gates = closure.loc[closure.source.eq('DATA')].set_index('plane')
    if set(gates.loc[gates.approved].index) != {'PP','PD'}:
        raise AssertionError('Only PP/PD were previously displayed in the reference inclusive DATA figure.')
    if not (finalize_selected_vr_displays_np.isclose(gates.tx,.3).all() and finalize_selected_vr_displays_np.isclose(gates.ty,2.6).all()):
        raise AssertionError('The previous inclusive reference gate must be (.3,2.6).')
    figures = finalize_selected_vr_displays_pd.read_csv(directory/'figure_manifest.csv')
    figure = figures.loc[figures.name.eq('mass_data_planes_Reference')]
    if len(figure) != 1:
        raise AssertionError('Require the existing inclusive reference DATA figure.')
    paths = [directory/name for name in ('manifest.json','mass_closure.csv','mass_distribution_bin_purity.csv',
                                        'mass_input_manifest.csv','figure_manifest.csv')]
    paths += [directory/figure.iloc[0].png]
    for path in paths:
        if not path.is_file():
            raise AssertionError('Missing previous inclusive DATA display evidence: '+str(path))
        inputs.append(finalize_selected_vr_displays_common.candidate_input_record(path, 'Previously public inclusive PP/PD exact folded DATA bins'))
    frame = finalize_selected_vr_displays_pd.read_csv(directory/'mass_distribution_bin_purity.csv', float_precision='round_trip')
    if len(frame) != 462 or not frame.signal_hypotheses.eq(120).all():
        raise AssertionError('Inclusive 2D purity requires 154 bins in each of PP/PD/DP and all 120 signals.')
    rows = []
    for row in frame.to_dict('records'):
        state = row['plane']
        rows.append(dict(row, scope='INCLUSIVE_'+state, mass_slice='all', ix=int(row['bin_x']), iy=int(row['bin_y']),
            xlow=row['x_low'], xhigh=row['x_high'], ylow=row['y_low'], yhigh=row['y_high'],
            physical_xlow=row['x_low'], physical_xhigh=finalize_selected_vr_displays_np.inf if finalize_selected_vr_displays_np.isclose(row['x_high'],2.) else row['x_high'],
            physical_ylow=-finalize_selected_vr_displays_np.inf if finalize_selected_vr_displays_np.isclose(row['y_low'],0.) else row['y_low'],
            physical_yhigh=finalize_selected_vr_displays_np.inf if finalize_selected_vr_displays_np.isclose(row['y_high'],finalize_selected_vr_displays_np.pi) else row['y_high'],
            bin_approved=bool(row['approved']), previously_displayed=bool(row['approved'] and gates.loc[state,'approved'])))
    validation.append(dict(scope='Inclusive previous bins',
        check='Reference PP/PD integrated gates, existing DATA figure and exact 154-bin all-120 purity records are verified', passed=True))
    return finalize_selected_vr_displays_pd.DataFrame(rows)


def finalize_selected_vr_displays__validate_data_production(config, inputs, validation):
    lowmid = finalize_selected_vr_displays_Path(config['vr2_output_dir'])/'input_manifest.csv'
    inclusive = finalize_selected_vr_displays_Path(config['comprehensive_dir'])/'mass_input_manifest.csv'
    left = finalize_selected_vr_displays_pd.read_csv(lowmid).query("source == 'DATA selected VR candidates'")
    right = finalize_selected_vr_displays_pd.read_csv(inclusive).query("source == 'DATA mass-inclusive PP/PD/DP only'")
    fields = ['path','size_bytes','sha256']
    finalize_selected_vr_displays_pd.testing.assert_frame_equal(left[fields].sort_values('path').reset_index(drop=True),
                                  right[fields].sort_values('path').reset_index(drop=True), check_dtype=False)
    if len(left) != 68:
        raise AssertionError('The 01/03/07 prior displays must use exactly the same 68 DATA files.')
    inputs.append(finalize_selected_vr_displays_common.candidate_input_record(lowmid,'Previously public low/intermediate DATA production'))
    validation.append(dict(scope='DATA production', check='01/03 and 07 source path, size and SHA256 match for all 68 DATA files', passed=True))


def finalize_selected_vr_displays__read_previous_bins(config, approved, dp_audit):
    allowed = {'VR1_DD','VR1_PP','VR1_PD','VR1_DP','VR2_PP','VR2_PD','VR2_DP','INCLUSIVE_PP','INCLUSIVE_PD','INCLUSIVE_DP'}
    if set(approved.scope) != allowed or len(approved) != 930 or not approved.complementary_pass.all():
        raise AssertionError('Only 909 exact previous public bins and 21 jointly audited DP bins are eligible.')
    requested_dp = approved.loc[approved.scope.eq('INCLUSIVE_DP')]
    audited_dp = dp_audit['tables']['dp_bin_audit'].loc[lambda frame:frame.accepted]
    finalize_selected_vr_displays__assert_dp_bin_equality(requested_dp, audited_dp)
    if not requested_dp.joint_release_pass.all() or not requested_dp.release_provenance.eq('NEW_JOINT_AUDITED_DP_BIN').all():
        raise AssertionError('Every new DP query requires its exact prospective joint-release approval.')
    bkg = finalize_selected_vr_displays_Path(config['bkg_dir'])
    paths = sorted(bkg.with_name(bkg.name.replace('_bkg_','_data_')).glob('*.coffea'))
    prior = finalize_selected_vr_displays_pd.read_csv(finalize_selected_vr_displays_Path(config['vr2_output_dir'])/'input_manifest.csv')
    prior = prior.loc[prior.source.eq('DATA selected VR candidates')].set_index('path')
    if len(paths) != 68 or set(map(str,paths)) != set(prior.index):
        raise AssertionError('The original DATA file set differs from the prior public production.')
    manifest = []
    for path in paths:
        record = finalize_selected_vr_displays_common.candidate_input_record(path,'909 exact previous 01/03/07 bins plus 21 explicitly joint-audited inclusive DP bins')
        expected = prior.loc[str(path)]
        if (record['sha256'],record['size_bytes']) != (expected.sha256,expected.size_bytes):
            raise AssertionError('Original DATA hash differs from the previous public production.')
        manifest.append(record)
    rows = approved.to_dict('records')
    counts = {(row['scope'],row['ix'],row['iy']):0. for row in rows}
    for index,path in enumerate(paths,1):
        loaded,payload = finalize_selected_vr_displays_common._extension_data_payload(path)
        weights,variances,edges = finalize_selected_vr_displays_common._comparison_current_cube(payload,path,config)
        for row in rows:
            if row['plane'] == 'DD':
                if row['mass_slice'] != 'low':
                    raise AssertionError('No nominal or intermediate/high-mass DD DATA query is permitted.')
                partner = slice(finalize_selected_vr_displays_common._comparison_split(edges['iso1'],0.),finalize_selected_vr_displays_common._comparison_split(edges['iso1'],.5))
                selection = [slice(None),partner,slice(None,finalize_selected_vr_displays_common._comparison_split(edges['disp0'],2.5)),
                    slice(finalize_selected_vr_displays_common._comparison_split(edges['disp1'],.5),None),slice(None)]
            else:
                selection = list(finalize_selected_vr_displays_buffered._state_index(edges,row['plane'],config))
            lo,hi = {'low':(0.,150.),'intermediate':(150.,300.),'all':(0.,finalize_selected_vr_displays_np.inf)}[row['mass_slice']]
            selection.append(slice(finalize_selected_vr_displays_common._comparison_split(edges['ljlj_mass'],lo),
                None if not finalize_selected_vr_displays_np.isfinite(hi) else finalize_selected_vr_displays_common._comparison_split(edges['ljlj_mass'],hi)))
            selection[0] = slice(finalize_selected_vr_displays_common._comparison_split(edges['iso0'],row['physical_xlow']),
                None if not finalize_selected_vr_displays_np.isfinite(row['physical_xhigh']) else finalize_selected_vr_displays_common._comparison_split(edges['iso0'],row['physical_xhigh']))
            selection[4] = slice(None if not finalize_selected_vr_displays_np.isfinite(row['physical_ylow']) else finalize_selected_vr_displays_common._comparison_split(edges['abs_dphi'],row['physical_ylow']),
                None if not finalize_selected_vr_displays_np.isfinite(row['physical_yhigh']) else finalize_selected_vr_displays_common._comparison_split(edges['abs_dphi'],row['physical_yhigh']))
            values,vs = weights[tuple(selection)],variances[tuple(selection)]
            if (not finalize_selected_vr_displays_np.isfinite(values).all() or (values<0).any() or not finalize_selected_vr_displays_np.array_equal(values,vs)
                    or not finalize_selected_vr_displays_np.array_equal(values,finalize_selected_vr_displays_np.rint(values))):
                raise AssertionError('Every approved DATA bin must have nonnegative integer counts and sumw2=N.')
            counts[(row['scope'],row['ix'],row['iy'])] += float(values.sum())
        del loaded,payload,weights,variances
        finalize_selected_vr_displays_gc.collect()
        if index%20 == 0 or index == len(paths):
            print(f'09 selected VR approved DATA bins {index}/{len(paths)}',flush=True)
    return counts,manifest


def finalize_selected_vr_displays__selected_scans(focused, latest):
    rows = []
    for study in ('loose','gap'):
        source = focused['if_vr_closure'] if study == 'loose' else latest['sv_closure'].loc[latest['sv_closure'].strategy.eq('gap')]
        spec = finalize_selected_vr_displays_SELECTIONS[study]
        axes = [('a_iso','a_phi',spec['a_phi'],spec['a_iso']),
                ('a_phi','a_iso',spec['a_iso'],spec['a_phi'])] if study == 'loose' else [
                ('b_iso','c_phi',spec['c_phi'],spec['b_iso']),
                ('c_phi','b_iso',spec['b_iso'],spec['c_phi'])]
        for axis,fixed,value,selected in axes:
            part = source.loc[finalize_selected_vr_displays_np.isclose(source[fixed],value)].copy()
            if axis in ('a_iso','b_iso'):
                part = part.loc[part[axis].between(.2,.5)]
            if axis == 'a_phi':
                part = part.loc[part[axis].between(2.4,2.8)]
            part['study'] = study
            part['scan_axis'] = axis
            part['scan_value'] = part[axis]
            part['fixed_axis'] = fixed
            part['fixed_value'] = value
            part['selected_point'] = finalize_selected_vr_displays_np.isclose(part[axis],selected)
            rows.append(part)
    result = finalize_selected_vr_displays_pd.concat(rows,ignore_index=True)
    withheld = result.source.eq('DATA') & ~result.display_allowed
    if not result.loc[withheld,['A','B','C','D','pred','kappa']].isna().all().all():
        raise AssertionError('All withheld 1D DATA closure values must remain NaN.')
    return result


def finalize_selected_vr_displays__assert_dp_bin_equality(requested, audited):
    fields = ['scope','plane','mass_slice','ix','iy','physical_xlow','physical_xhigh',
              'physical_ylow','physical_yhigh']
    left = requested[fields].sort_values(['ix','iy']).reset_index(drop=True)
    right = audited[fields].sort_values(['ix','iy']).reset_index(drop=True)
    finalize_selected_vr_displays_pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=True)


def finalize_selected_vr_displays__apply_dp_audit(bins, result, proof, load_data):
    metadata = result['metadata']
    if (metadata['original_data_files_opened'] != 0 or metadata['accepted_bins'] != 21
            or metadata['new_linear_functionals'] != 20
            or metadata['newly_identifiable_unsafe_targets'] != 0
            or not metadata['independent_final_QR_pass']):
        raise AssertionError('The prospective DP release must satisfy its full final joint audit.')
    audit = result['tables']['dp_bin_audit']
    if len(audit) != 77 or not audit.bin_purity_pass.all() or not audit.signal_hypotheses.eq(120).all():
        raise AssertionError('Audit exactly the 77 MC-pure inclusive DP display bins with all 120 signals.')
    accepted = audit.loc[audit.accepted]
    if len(accepted) != 21 or accepted.rank_increment.sum() != 20:
        raise AssertionError('The accepted DP whitelist or independent-query count changed.')
    if accepted.unsafe_exhaustive.any() or accepted.unsafe_joint.any():
        raise AssertionError('No accepted DP bin may fail either complementary protection family.')
    inclusive = bins.loc[bins.scope.eq('INCLUSIVE_DP')]
    if inclusive.release_eligible.any() or inclusive.previously_displayed.any():
        raise AssertionError('Inclusive DP has no previously displayed bin; a new bin requires the explicit audit.')
    for row in audit.to_dict('records'):
        selected = inclusive.loc[inclusive.ix.eq(row['ix']) & inclusive.iy.eq(row['iy'])]
        if len(selected) != 1:
            raise AssertionError('Each prospective DP bin must identify one physical display cell.')
        finalize_selected_vr_displays__assert_dp_bin_equality(selected,finalize_selected_vr_displays_pd.DataFrame([row]))
        actual = selected.iloc[0]
        finalize_selected_vr_displays_np.testing.assert_allclose([actual.background,actual.max_S_B],
            [row['background'],row['max_S_B']],rtol=1e-10,atol=1e-10)
        index = selected.index[0]
        if not row['accepted']:
            bins.loc[index,'reason'] = row['reason']
            continue
        bins.loc[index,['complementary_pass','joint_release_pass','release_eligible','display_allowed']] = [True,True,True,bool(load_data)]
        bins.loc[index,'release_provenance'] = 'NEW_JOINT_AUDITED_DP_BIN'
        bins.loc[index,'already_identifiable'] = bool(row['already_identifiable'])
        bins.loc[index,'rank_increment'] = int(row['rank_increment'])
        bins.loc[index,'reason'] = 'ALL120_PURITY_AND_PROSPECTIVE_JOINT_NATIVE_RECTANGLE_MASS_AUDIT_PASS'
        proof.append(dict(scope=row['scope'],plane='DP',mass_slice='all',ix=row['ix'],iy=row['iy'],
            exact_previous_mask=False,source_notebook='',newly_queried_display_bin=True,
            already_identifiable=bool(row['already_identifiable']),
            newly_released_linear_functionals=int(row['rank_increment']),
            newly_identifiable_unsafe_regions=0,
            reason='Prospective sequential joint/complement and native-rectangle/mass-union audit plus independent final QR verification.'))
    finalize_selected_vr_displays__assert_dp_bin_equality(bins.loc[bins.scope.eq('INCLUSIVE_DP') & bins.release_eligible],accepted)


def finalize_selected_vr_displays__selected_closure_gate(study,spec,scope,mass,focused,latest):
    if mass == 'nominal':
        return False,'NOMINAL_DATA_ALWAYS_BLINDED'
    if study == 'loose':
        source = focused['if_vr_closure']
        selected = source.source.eq('DATA') & source.scope.eq(scope)
        for field in ('a_iso','a_phi','b_iso','c_phi'):
            selected &= finalize_selected_vr_displays_np.isclose(source[field],spec[field],rtol=0,atol=1e-12)
        release = source.loc[selected]
        if len(release) != 1:
            raise AssertionError('Selected loose closure requires one exact focused geometry and public-count reuse decision.')
        row = release.iloc[0]
        if row.display_allowed and (not row.approved or row.signal_hypotheses != 120
                or not finalize_selected_vr_displays_np.isfinite(row.max_S_B) or row.max_S_B >= .1
                or not bool(row.get('reused_previous_data',False))
                or bool(row.get('new_data_evaluated',False))
                or not finalize_selected_vr_displays_np.isfinite(row[list('ABCD')].to_numpy(float)).all()):
            raise AssertionError('A reused loose closure must retain current all-region purity and finite previously public ABCD counts.')
    else:
        source = latest['sv_release_audit']
        release = source.loc[source.scope.eq(scope) & source.candidate_id.eq(spec['candidate_id'])]
        if len(release) != 1:
            raise AssertionError('The selected buffer requires one exact current regional-public-reuse gate.')
        row = release.iloc[0]
    return bool(row.display_allowed),str(row.reason)


def finalize_selected_vr_displays_prepare_final_gap_vr(config, latest_scan):
    """Reuse exact public native-grid regions for the final capped gap scan.

    The frozen preceding report remains untouched.  A gap's A/B/C/D rectangles
    have the same masks as individual regions at four ordinary native-grid
    coordinates; reusing these public counts introduces no new observation.
    A historical subset-audit veto cannot retract measurements already public
    in 01/03.  Current all-region/all-120 contamination remains mandatory.
    """
    _validate_public_vr_grids = _FINALIZE_NAMESPACES['abcd_iso_focus']._validate_public_vr_grids
    _load_caches = _FINALIZE_NAMESPACES['abcd_sideband_validation']._load_caches; MEASUREMENT_FIELDS = _FINALIZE_NAMESPACES['abcd_sideband_validation'].MEASUREMENT_FIELDS
    _cached_sources = _FINALIZE_NAMESPACES['abcd_sideband_scan_validation']._cached_sources; data_statistics = _FINALIZE_NAMESPACES['abcd_sideband_scan_validation'].data_statistics
    region_rectangles = _FINALIZE_NAMESPACES['abcd_sideband_tradeoff'].region_rectangles

    latest = latest_scan.get('tables', latest_scan)
    closure = latest['sv_closure'].copy(deep=True)
    releases = latest['sv_release_audit'].copy(deep=True)
    inputs, validation, proofs = [], [], []
    caches = _load_caches(config, inputs, validation)
    inclusive = finalize_selected_vr_displays_pd.read_csv(finalize_selected_vr_displays_Path(config['comprehensive_dir'])/'mass_grid.csv', float_precision='round_trip')
    _validate_public_vr_grids(config, caches, inclusive, inputs, validation)
    selected = closure.source.eq('DATA') & closure.strategy.eq('gap') & closure.b_iso.between(.2,.5)
    selected &= closure.mass_slice.isin(['low','intermediate'])
    restored = 0
    for index, row in closure.loc[selected].iterrows():
        if not bool(row.approved) or not finalize_selected_vr_displays_np.isfinite(row.max_S_B) or row.max_S_B >= .1:
            continue
        available, source = _cached_sources(caches, row.mass_slice, row.plane, row)
        if not available:
            continue
        mc = closure.loc[closure.source.eq('MC') & closure.scope.eq(row.scope)
                         & closure.candidate_id.eq(row.candidate_id)]
        purity = latest['sv_purity'].loc[latest['sv_purity'].scope.eq(row.scope)
                    & latest['sv_purity'].candidate_id.eq(row.candidate_id)]
        if len(mc) != 1 or len(purity) != 4 or not purity.approved.all() or not purity.signal_hypotheses.eq(120).all():
            raise AssertionError('Final buffer closure requires one MC row and all four current all-120 purity passes.')
        mc = mc.iloc[0]
        values, variances = {}, {}
        for region, (x, y) in finalize_selected_vr_displays_candidate_coordinates(row).items():
            original = source[region]
            ordinary = dict(a_iso=x, a_phi=y, b_iso=x, c_phi=y)
            if region_rectangles(row)[region] != region_rectangles(ordinary)[region]:
                raise AssertionError('Reused source DATA must equal the complete requested physical region.')
            original_mc = caches[row.mass_slice]['grid']
            original_mc = original_mc.loc[original_mc.source.eq('MC') & original_mc.plane.eq(row.plane)
                            & finalize_selected_vr_displays_np.isclose(original_mc.tx,x) & finalize_selected_vr_displays_np.isclose(original_mc.ty,y)]
            if len(original_mc) != 1 or original.signal_hypotheses != 120 or not original.approved:
                raise AssertionError('Require a unique previously approved native DATA coordinate and matching MC source.')
            finalize_selected_vr_displays_np.testing.assert_allclose([mc[region],mc['var_'+region]],
                [original_mc.iloc[0][region],original_mc.iloc[0]['var_'+region]],rtol=1e-8,atol=1e-8)
            values[region], variances[region] = float(original[region]), float(original['var_'+region])
            proofs.append(dict(candidate_id=row.candidate_id, scope=row.scope, plane=row.plane,
                mass_slice=row.mass_slice, region=region, source_notebook='01' if row.mass_slice=='low' else '03',
                source_tx=x,source_ty=y,exact_physical_region=True,source_approved=True,
                signal_hypotheses=120, value=values[region],variance=variances[region],
                original_report_display_allowed=bool(row.display_allowed),new_independent_observations=0,
                reason='EXACT_PREVIOUSLY_PUBLIC_NATIVE_GRID_REGION_REUSE'))
        stat = data_statistics(values, variances, config.get('min_neff',10.))
        if row.display_allowed:
            finalize_selected_vr_displays_np.testing.assert_allclose([row[field] for field in MEASUREMENT_FIELDS],
                [stat[field] for field in MEASUREMENT_FIELDS],rtol=1e-10,atol=1e-10,equal_nan=True)
        else:
            restored += 1
        for field,value in stat.items():
            closure.loc[index,field] = value
        closure.loc[index,['display_allowed','release_eligible']] = True
        closure.loc[index,'reason'] = 'PREVIOUSLY_PUBLIC_NATIVE_REGIONS_EXACT_REUSE'
        match = releases.scope.eq(row.scope) & releases.candidate_id.eq(row.candidate_id)
        if match.sum() != 1:
            raise AssertionError('Every final buffer row needs one matching release record.')
        for field in ('complementary_pass','release_eligible','display_allowed','source_cache_approved'):
            releases.loc[match,field] = True
        releases.loc[match,'reason'] = 'PREVIOUSLY_PUBLIC_NATIVE_REGIONS_EXACT_REUSE'
        releases.loc[match,['new_complement_checks','failed_complement_checks']] = 0
    hidden = closure.source.eq('DATA') & ~closure.display_allowed
    if not closure.loc[hidden,MEASUREMENT_FIELDS].isna().all().all():
        raise AssertionError('Contaminated or unsupported final buffer DATA rows must remain NaN.')
    validation.extend([
        dict(scope='Final buffer public counts',check='Every reused A/B/C/D equals one already public 01/03 native-grid physical region; no new independent DATA information',passed=True),
        dict(scope='Final buffer current purity',check='Only current all-four-region, all-120 strict MC contamination passes are displayed',passed=True),
        dict(scope='Final buffer blinding',check='No original DATA file, nominal DATA A, high-mass DD, or new inclusive integral is accessed',passed=True),
    ])
    return dict(tables=dict(sv_closure=closure,sv_release_audit=releases,
        fgvr_reuse_audit=finalize_selected_vr_displays_pd.DataFrame(proofs),fgvr_validation=finalize_selected_vr_displays_pd.DataFrame(validation),
        fgvr_input_manifest=finalize_selected_vr_displays_pd.DataFrame(inputs).drop_duplicates(['path','sha256'])),
        metadata=dict(restored_gap_DATA_rows=restored,exact_public_regional_queries=len(proofs),
            new_independent_DATA_observations=0,original_DATA_files_opened=0,
            current_all_ABCD_contamination_required=True,nominal_A_evaluated=False))
