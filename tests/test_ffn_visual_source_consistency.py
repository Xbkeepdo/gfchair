import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from features.ffn_visual_path_attribution import streaming_vector_path_statistics
from scripts.run_ffn_visual_source_consistency import dual_net, local_fp32, immutable_json, select_cases, audit, VERSION
from scripts.analyze_ffn_visual_source_consistency import (
    fit_scales, feature_sets, FEATURE_GROUPS, gate_group, compare_measurement,
    assert_cohort_isolation, select_primary, checkpoint_probabilities,
)


class ConsistencyTest(unittest.TestCase):
    def test_training_only_queue_never_extracts_freezes_or_generates(self):
        from scripts.analyze_ffn_visual_source_consistency import pipeline,MODELS
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); commands=[]
            def run(command,**kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0)
            args=SimpleNamespace(stage='train-only',config='unused',train_despite_known_failure=True,
                                 production_after_subset=True,full_reference=False)
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',root), \
                 patch('scripts.analyze_ffn_visual_source_consistency.result_root',side_effect=lambda m:root/m), \
                 patch('subprocess.run',side_effect=run):
                pipeline(args)
            stages=[c[3] for c in commands]
            self.assertEqual(stages.count('train'),4)
            self.assertEqual(stages.count('verify-detectors'),4)
            self.assertEqual(stages.count('summarize'),1)
            self.assertEqual(len(stages),9)
            self.assertTrue(all('--train-despite-known-failure' in c for c in commands))

    def test_training_exception_retains_fail_and_rejects_changed_gate(self):
        from scripts.analyze_ffn_visual_source_consistency import training_numerical_exception,load_training_data,MODELS
        from features.tc_fvpa_artifacts import atomic_json_save,sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'outputs'/VERSION/'old_validation_fp32_k4.json'
            models={m:dict(processed_images=4000,layers={'23':dict(passed=True,closure_max=.001)}) for m in MODELS}
            models['qwen2_5_vl_7b']['layers']['23'].update(passed=False,closure_max=.010615984949452763)
            value=dict(status='FAIL_NUMERICAL_OLD_COHORT',candidate='fp32_k4',models=models)
            atomic_json_save(value,path); before=(sha256_file(path),path.stat().st_mtime_ns)
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',root):
                with self.assertRaisesRegex(ValueError,'acceptance must precede training'):
                    load_training_data('qwen2_5_vl_7b','fp32_k4')
                with self.assertRaises(FileNotFoundError):
                    training_numerical_exception(path)
                exception=training_numerical_exception(path,create=True)
                self.assertFalse(exception['independent_confirmation_authorized'])
                self.assertEqual(exception['numerical_validation_status'],'FAIL_NUMERICAL_OLD_COHORT')
                self.assertEqual(training_numerical_exception(path),exception)
                self.assertEqual(before,(sha256_file(path),path.stat().st_mtime_ns))
                value['modified']=True; atomic_json_save(value,path)
                with self.assertRaisesRegex(ValueError,'exception fingerprint'):
                    training_numerical_exception(path)
                value['models']['qwen3_vl_8b']['layers']['23']['passed']=False; atomic_json_save(value,path)
                with self.assertRaisesRegex(ValueError,'does not cover'):
                    training_numerical_exception(path,create=True)

    def test_tail_diagnostic_snapshot_and_reproduction(self):
        from scripts.diagnose_ffn_visual_source_tail import snapshot, reproduction
        torch.manual_seed(11)
        z, writes = torch.randn(2,5), .05*torch.randn(9,2,5)
        rows = []
        for k in (4,64):
            stats = streaming_vector_path_statistics(ffn_map=torch.sin,z=z,writes=writes,
                method='gauss_legendre',integration_points=k,token_chunk_size=3,save_components=True)
            row = snapshot(stats,1)
            torch.testing.assert_close(row['components'].sum(0),row['component_sum'])
            self.assertAlmostEqual(row['N_vec'],float(row['component_sum'].norm()))
            self.assertTrue(reproduction(row,row)['passed'])
            self.assertFalse(reproduction(row,dict(row,S=2*row['S']))['passed'])
            self.assertFalse(reproduction(row,dict(row,closure_relative_error=.02))['passed'])
            rows.append(row)
        self.assertTrue(torch.equal(rows[0]['endpoint'],rows[1]['endpoint']))
        self.assertLess(compare_measurement(*rows)['S_relative'],1e-5)

    def test_dual_net_linear_nonlinear_and_chunks(self):
        torch.manual_seed(7)
        z, writes = torch.randn(2,5), .1*torch.randn(9,2,5)
        for function in (lambda x: 2*x, torch.sin):
            stats = []
            for chunk in (3,9):
                r = streaming_vector_path_statistics(ffn_map=function,z=z,writes=writes,
                    method='gauss_legendre',integration_points=64,token_chunk_size=chunk,save_components=True)
                torch.testing.assert_close(r.components.sum(0),r.component_sum)
                for i in range(2):
                    d = dual_net(r.component_sum[i],r.total_finite_effect[i],r.gross_strength[i])
                    self.assertTrue(d['kappa_range_pass'] and d['kappa_difference_bound_pass'] and d['closure_lower_bound_pass'])
                    self.assertLess(d['closure_relative_error'], 1e-5)
                stats.append(r)
            torch.testing.assert_close(stats[0].component_sum,stats[1].component_sum)
            torch.testing.assert_close(stats[0].p_ffn,stats[1].p_ffn)

    def test_zero_strength_and_endpoint(self):
        d = dual_net(torch.zeros(3),torch.zeros(3),0.)
        self.assertEqual(d['kappa_vec'],0)
        self.assertIsNone(d['closure_relative_error'])
        self.assertTrue(d['endpoint_zero'])
        d = dual_net(torch.ones(3),torch.zeros(3),2.)
        self.assertIsNone(d['closure_relative_error'])
        self.assertGreater(d['closure_absolute_error'],0)
        with self.assertRaises(ValueError):
            dual_net(torch.tensor([float('nan')]),torch.ones(1),1.)

    def test_endpoint_kappa_is_not_clipped(self):
        d = dual_net(torch.tensor([.5]),torch.tensor([3.]),1.)
        self.assertEqual(d['kappa_end'],3.)
        self.assertEqual(d['kappa_vec'],.5)
        self.assertTrue(d['kappa_difference_bound_pass'] and d['closure_lower_bound_pass'])

    def test_dtype_and_tf32_restore_on_failure(self):
        adapter = SimpleNamespace(ffn_norm=nn.LayerNorm(3).bfloat16(),ffn=nn.Linear(3,3).bfloat16(),
                                  output_projection=nn.Linear(3,3).bfloat16())
        original = [p.clone() for m in (adapter.ffn_norm,adapter.ffn) for p in m.parameters()]
        old = torch.backends.cuda.matmul.allow_tf32
        with patch('scripts.run_ffn_visual_source_consistency.resolve_decoder_layer_adapter',return_value=adapter):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with local_fp32(None) as fn:
                    self.assertEqual(fn(torch.ones(1,3)).dtype,torch.float32)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    raise RuntimeError('injected')
        self.assertEqual(old,torch.backends.cuda.matmul.allow_tf32)
        actual = [p for m in (adapter.ffn_norm,adapter.ffn) for p in m.parameters()]
        for a,b in zip(original,actual):
            self.assertEqual(b.dtype,torch.bfloat16)
            torch.testing.assert_close(a,b)

    def test_train_only_common_scale_and_raw_fusion(self):
        S = np.array([[2.,4.],[4.,8.],[10000.,20000.]])
        scales = fit_scales(S,S*3,[1,2,3],{1,2})
        np.testing.assert_array_equal(scales['S'],[3.,6.])
        np.testing.assert_array_equal(scales['I'],[9.,18.])
        raw = {k:np.ones((3,2)) for k in ('AE','R_cos','D_EW','D_WF','D_EF','kappa_vec','kappa_end')}
        raw.update(S=S,N_vec=S/4,N_end=S/3,I=S*3)
        features = feature_sets(raw,scales)
        self.assertEqual(list(features),list(FEATURE_GROUPS))
        self.assertEqual(len(features),16)
        np.testing.assert_allclose(features['AE+a'][:,2:],np.log1p((S+S/4)/2/scales['S']),rtol=1e-6)
        np.testing.assert_allclose(features['AE+g'][:,2:],np.log1p(S/2/scales['S']),rtol=1e-6)
        self.assertFalse(np.allclose(features['AE+a'][:,2:],(np.log1p(S/scales['S'])+np.log1p(S/4/scales['S']))/2))
        for name,order in FEATURE_GROUPS.items():
            self.assertEqual(features[name].shape,(3,2*len(order)))
        np.testing.assert_array_equal(features['H_SN'][:,:4],features['U_SN'][:,:4])
        np.testing.assert_array_equal(features['H_SN'][:,-4:],features['U_SN'][:,-4:])

    def test_immutable_resume_rejects_changes_without_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            immutable_json({'fingerprint':'a'},path)
            before = path.stat().st_mtime_ns
            immutable_json({'fingerprint':'a'},path)
            self.assertEqual(before,path.stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError,'Fingerprint'):
                immutable_json({'fingerprint':'b'},path)
            self.assertEqual(json.loads(path.read_text()),{'fingerprint':'a'})

    def test_analysis_json_resume_normalizes_integer_keys_and_tuples(self):
        from scripts.analyze_ffn_visual_source_consistency import immutable_json as save
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'validation.json'
            value={'layers':{1:{'pass':True}},'order':('S','N_vec')}
            save(value,path)
            before=path.stat().st_mtime_ns
            save(value,path)
            self.assertEqual(before,path.stat().st_mtime_ns)
            with self.assertRaises(ValueError):
                save({'nonfinite':float('nan')},Path(directory)/'invalid.json')

    def test_audit_rejects_wrong_shard_fingerprint_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            case=dict(image_id=1,response_index=0,layer=1,case_id='1:0:1',groups=['all_layers'])
            immutable_json(dict(model='qwen2_5_vl_7b',version=VERSION,cases=[case],inputs={}),root/'cases.json')
            path=root/'shards/audit/case_1_0_1.pt'
            path.parent.mkdir(parents=True)
            torch.save(dict(fingerprint='wrong',case=case),path)
            args=SimpleNamespace(model='qwen2_5_vl_7b',output_root=str(root),cohort_manifest=str(root/'cases.json'),
                config='configs/model_configs_inslen_official_target.yaml',token_chunk_size=256,rank=0,world_size=1,smoke=False,resume=True)
            with patch('scripts.run_ffn_visual_source_consistency.build_model',side_effect=AssertionError('model must not load')):
                with self.assertRaisesRegex(ValueError,'Wrong resume fingerprint'):
                    audit(args)

    def test_stratified_anomaly_and_nearest_unique_controls(self):
        rows = [dict(image_id=i,response_index=0,layer=1,label=0,old_S=float(i),
                     visual_token_count=32,old_kappa_end=2. if i in (2,3) else .5,
                     old_closure_relative_error=i*.1) for i in range(1,7)]
        selected = select_cases(rows,limit=2)
        self.assertEqual(len(selected),4)
        normal = [r for r in selected if r['groups']==['normal']]
        self.assertEqual(len({r['image_id'] for r in normal}),2)
        self.assertEqual([r['image_id'] for r in normal],[1,4])

    def test_gate_rejects_error_and_nonfinite_comparisons(self):
        d = dual_net(torch.ones(3),torch.ones(3),3.)
        value = dict(d,p_ffn=np.ones(32)/32)
        d.update(compare_measurement(value,value))
        self.assertEqual(gate_group([d])['status'],'PASS')
        d['closure_relative_error']=.011
        self.assertEqual(gate_group([d])['status'],'FAIL')
        d['closure_relative_error']=1e-5
        d['N_vec_relative']=None
        self.assertEqual(gate_group([d])['status'],'FAIL')

    def test_cohort_ids_and_checksums_are_both_disjoint(self):
        assert_cohort_isolation([dict(image_id=2,sha256='b')],{1},{'a'})
        for selected in ([dict(image_id=1,sha256='b')],[dict(image_id=2,sha256='a')],
                         [dict(image_id=2,sha256='b'),dict(image_id=3,sha256='b')]):
            with self.assertRaises(ValueError):
                assert_cohort_isolation(selected,{1},{'a'})

    def test_fusion_selection_requires_both_tolerances_for_every_model(self):
        def summary(auroc,ap):
            return dict(ensemble_reports={'fixed_0.5':dict(auc=auroc,hallucination_positive={'aupr':ap})})
        values = {str(i):dict(groups={'AE+s+n':summary(.9,.7),'AE+a':summary(.899,.695),'AE+g':summary(.899,.699)}) for i in range(4)}
        self.assertEqual(select_primary(values)[0],'AE+a')
        values['3']['groups']['AE+a']=summary(.899,.68)
        self.assertEqual(select_primary(values)[0],'AE+g')
        values['2']['groups']['AE+g']=summary(.89,.7)
        self.assertEqual(select_primary(values)[0],'AE+s+n')

    def test_checkpoint_cpu_prediction(self):
        from scripts.train_torch_probe_feature_sets import DGSTStyleProbe
        model = DGSTStyleProbe(4,(128,64,32),.3).eval()
        matrix = np.ones((17,4),dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(model.state_dict(),path)
            with torch.no_grad():
                expected=torch.sigmoid(model(torch.from_numpy(matrix))).numpy().reshape(-1)
            np.testing.assert_allclose(checkpoint_probabilities(path,matrix),expected,rtol=1e-6)

    def test_saved_trajectories_match_dual_records_and_zero_target_images(self):
        from scripts.analyze_ffn_visual_source_consistency import check_extracted_image
        manifest=dict(fingerprint='verified')
        row=dict(processed_image=True,image_ids=[1],fingerprint='verified',sample_table=[],positions=[])
        check_extracted_image(row,manifest,1,'qwen2_5_vl_7b',[],[])
        value=dual_net(torch.ones(2),torch.ones(2),2.)
        position={k:np.ones(28,dtype=np.float32) for k in ('AE','I','R_cos','D_EW','D_WF','D_EF')}
        position.update({k:np.full(28,value[k],dtype=np.float32) for k in ('S','N_vec','N_end','kappa_vec','kappa_end')})
        position.update(dual_net=[value]*28,response_index=0)
        row['positions']=[position]
        check_extracted_image(row,manifest,1,'qwen2_5_vl_7b',[],[0])
        position['N_vec'][5]=0
        with self.assertRaisesRegex(ValueError,'Dual-net scalar mismatch'):
            check_extracted_image(row,manifest,1,'qwen2_5_vl_7b',[],[0])

    def test_independent_generation_requires_freeze_before_model_load(self):
        from scripts.analyze_ffn_visual_source_consistency import generate_new
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',Path(directory)):
                with patch('models.build_model',side_effect=AssertionError('must not load')):
                    with self.assertRaises(FileNotFoundError):
                        generate_new(SimpleNamespace())

    def test_all_heads_tiny_training_checkpoint_metrics_and_no_retrain_resume(self):
        from scripts.analyze_ffn_visual_source_consistency import train,verify_detectors
        from scripts.train_torch_probe_feature_sets import TorchProbeConfig
        from features.tc_fvpa_artifacts import sha256_file
        rng=np.random.default_rng(9)
        data=dict(X_train={k:rng.normal(size=(24,2*len(v))).astype(np.float32) for k,v in FEATURE_GROUPS.items()},
                  X_test={k:rng.normal(size=(12,2*len(v))).astype(np.float32) for k,v in FEATURE_GROUPS.items()},
                  y_train=np.tile([0,1],12).astype(np.int32),y_test=np.tile([0,1],6).astype(np.int32),
                  scales={'S':np.ones(2),'I':np.ones(2)},train_mentions=[{'image_id':i} for i in range(24)],
                  test_mentions=[{'image_id':i} for i in range(24,36)],validation_sha256='synthetic-test-only',
                  numerical_exception={'status':'synthetic-exploratory-exception','sha256':'test-only'})
        args=SimpleNamespace(model='qwen2_5_vl_7b',candidate='fp32_k4',device='cpu',train_despite_known_failure=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('scripts.analyze_ffn_visual_source_consistency.load_training_data',return_value=data), \
                 patch('scripts.analyze_ffn_visual_source_consistency.result_root',return_value=root), \
                 patch('scripts.train_torch_probe_feature_sets.TorchProbeConfig',side_effect=lambda seed=43:TorchProbeConfig(seed=seed,num_epochs=2)), \
                 patch('builtins.print'):
                train(args)
                verify_detectors(args)
                paths=[p for p in root.rglob('*') if p.is_file()]
                before={str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in paths}
                with patch('scripts.train_torch_probe_feature_sets.train_and_evaluate_probe',side_effect=AssertionError('must not retrain')):
                    train(args)
                    verify_detectors(args)
                self.assertEqual(before,{str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in paths})
                self.assertEqual(len(list(root.rglob('model.pt'))),48)

    def test_pipeline_hard_stops_after_highest_numerical_failure(self):
        from scripts.analyze_ffn_visual_source_consistency import pipeline
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            global_root=root/'outputs'/VERSION
            global_root.mkdir(parents=True)
            immutable_json(dict(status='PASS_AUDIT',selected_candidate='native_k4'),global_root/'audit_gate.json')
            for name in ('qwen2_5_vl_7b','qwen3_vl_8b','llava_1_5_7b','internvl_2_5_8b'):
                (root/name).mkdir()
            commands=[]
            def run(command,**kwargs):
                commands.append(command)
                if command[3]=='validate-old':
                    candidate=command[command.index('--candidate')+1]
                    immutable_json(dict(status='FAIL_NUMERICAL_OLD_COHORT'),global_root/f'old_validation_{candidate}.json')
                return SimpleNamespace(returncode=0)
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',root), \
                 patch('scripts.analyze_ffn_visual_source_consistency.result_root',side_effect=lambda m:root/m), \
                 patch('subprocess.run',side_effect=run):
                with self.assertRaisesRegex(ValueError,'Highest numerical candidate failed'):
                    pipeline(SimpleNamespace(config='unused',full_reference=True))
            self.assertEqual(sum(c[3]=='extract-old' for c in commands),12)
            self.assertFalse(any(c[3] in ('train','freeze','generate-new') for c in commands))

    def test_subset_stops_at_boundary_and_resume_does_not_reextract(self):
        from scripts.analyze_ffn_visual_source_consistency import extract_subset
        from features.tc_fvpa_artifacts import atomic_torch_save,sha256_file
        import scripts.run_jffn_p_comparison as old
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model_root=root/'model'; cohort=model_root/'old/fp32_k4'
            global_root=root/'outputs'/VERSION/'numerical_subset_20260907'
            immutable_json(dict(fingerprint='original',image_ids=[1,2,3]),cohort/'manifest.json')
            immutable_json(dict(models={'llava_1_5_7b':dict(image_ids=[1,2],extraction_root=str(cohort),
                manifest_sha256=sha256_file(cohort/'manifest.json'))}),global_root/'selection.json')
            calls=[]
            def compute(**kwargs):
                calls.append(kwargs['image_id'])
            def full(args):
                for image_id in (1,2,3):
                    old._extract_one_image(image_id=image_id)
                    path=cohort/'shards'/f'image_{image_id:012d}.pt'
                    atomic_torch_save({'image_id':image_id},path)
                    immutable_json(dict(sha256=sha256_file(path),fingerprint='original'),path.with_suffix('.sha256.json'))
            args=SimpleNamespace(model='llava_1_5_7b',rank=0,world_size=1)
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',root), \
                 patch('scripts.analyze_ffn_visual_source_consistency.result_root',return_value=model_root), \
                 patch('scripts.run_jffn_p_comparison._extract_one_image',side_effect=compute), \
                 patch('scripts.analyze_ffn_visual_source_consistency.extract_old',side_effect=full):
                extract_subset(args)
                extract_subset(args)
            self.assertEqual(calls,[1,2])
            self.assertEqual(len(list((cohort/'shards').glob('*.pt'))),2)

    def test_subset_two_ranks_resume_without_overlap_or_boundary_overrun(self):
        from scripts.analyze_ffn_visual_source_consistency import extract_subset
        from features.tc_fvpa_artifacts import atomic_torch_save,sha256_file
        import scripts.run_jffn_p_comparison as old
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model_root=root/'model'; cohort=model_root/'old/fp32_k4'
            global_root=root/'outputs'/VERSION/'numerical_subset_20260907'
            immutable_json(dict(fingerprint='original',image_ids=list(range(1,9))),cohort/'manifest.json')
            immutable_json(dict(models={'llava_1_5_7b':dict(image_ids=list(range(1,6)),extraction_root=str(cohort),
                manifest_sha256=sha256_file(cohort/'manifest.json'))}),global_root/'selection.json')
            def save(image_id):
                path=cohort/'shards'/f'image_{image_id:012d}.pt'
                atomic_torch_save({'image_id':image_id},path)
                immutable_json(dict(sha256=sha256_file(path),fingerprint='original'),path.with_suffix('.sha256.json'))
            save(1); save(2)
            original_stats={p:(sha256_file(p),p.stat().st_mtime_ns) for p in (cohort/'shards').iterdir()}
            calls=[]
            def compute(**kwargs):
                calls.append(kwargs['image_id'])
            def full(args):
                for image_id in list(range(1,9))[args.rank::args.world_size]:
                    if (cohort/'shards'/f'image_{image_id:012d}.pt').exists():
                        continue
                    old._extract_one_image(image_id=image_id)
                    save(image_id)
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',root), \
                 patch('scripts.analyze_ffn_visual_source_consistency.result_root',return_value=model_root), \
                 patch('scripts.run_jffn_p_comparison._extract_one_image',side_effect=compute), \
                 patch('scripts.analyze_ffn_visual_source_consistency.extract_old',side_effect=full):
                for _ in range(2):
                    for rank in (0,1):
                        extract_subset(SimpleNamespace(model='llava_1_5_7b',rank=rank,world_size=2))
                extract_subset(SimpleNamespace(model='llava_1_5_7b',rank=0,world_size=1))
            self.assertEqual(calls,[3,5,4])
            self.assertEqual(len(list((cohort/'shards').glob('*.pt'))),5)
            for path,stats in original_stats.items():
                self.assertEqual(stats,(sha256_file(path),path.stat().st_mtime_ns))
            self.assertTrue((global_root/'llava_1_5_7b_extraction.json').exists())
            for rank in (0,1):
                self.assertTrue((global_root/f'llava_1_5_7b_extraction_rank{rank}_of2.json').exists())

    def test_production_reuse_preserves_originals_and_records_ancestry(self):
        import inspect
        from scripts.analyze_ffn_visual_source_consistency import seed_production,extract_old
        from features.tc_fvpa_artifacts import atomic_torch_save,sha256_file,sha256_text
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; production=root/'production'
            global_root=root/'outputs'/VERSION/'numerical_subset_20260907'
            source_manifest=dict(candidate='fp32_k4',full_k64_reference=True,
                implementation={'extract_old':sha256_text(inspect.getsource(extract_old))},
                fingerprint='old',image_ids=[1,2])
            immutable_json(source_manifest,source/'manifest.json')
            path=source/'shards/image_000000000001.pt'
            atomic_torch_save(dict(fingerprint='old',S=torch.tensor([3.]),N_vec=torch.tensor([2.])),path)
            immutable_json(dict(sha256=sha256_file(path),fingerprint='old'),path.with_suffix('.sha256.json'))
            immutable_json(dict(models={'qwen2_5_vl_7b':dict(extraction_root=str(source),image_ids=[1])}),global_root/'selection.json')
            immutable_json(dict(status='PASS_K4_NUMERICAL_SUBSET',models={'qwen2_5_vl_7b':dict(artifacts={str(path):sha256_file(path)})}),global_root/'gate.json')
            before=(sha256_file(path),path.stat().st_mtime_ns)
            with patch('scripts.analyze_ffn_visual_source_consistency.ROOT',root), \
                 patch('scripts.analyze_ffn_visual_source_consistency.result_root',return_value=production):
                seed_production()
                dest=production/'old/fp32_k4/shards'/path.name
                copy_before=(sha256_file(dest),dest.stat().st_mtime_ns)
                seed_production()
            self.assertEqual(before,(sha256_file(path),path.stat().st_mtime_ns))
            self.assertEqual(copy_before,(sha256_file(dest),dest.stat().st_mtime_ns))
            copied=torch.load(dest,weights_only=False)
            original=torch.load(path,weights_only=False)
            torch.testing.assert_close(copied['S'],original['S'])
            torch.testing.assert_close(copied['N_vec'],original['N_vec'])
            self.assertEqual(copied['reused_from']['sha256'],before[0])
            self.assertTrue(copied['reused_from']['original_included_k64_reference'])


if __name__=='__main__':
    unittest.main()
