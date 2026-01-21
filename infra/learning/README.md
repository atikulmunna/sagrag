Continuous learning exports

This folder provides a scheduled export of training data from the audit/feedback store.

Apply (Kubernetes)

1) Create the PVC:
   kubectl apply -f infra/learning/pvc.yaml
2) Create the CronJob:
   kubectl apply -f infra/learning/cron.yaml

Artifacts are written to `/data/learning/train.jsonl` in the `sag-rag-data` PVC.

Training stub (placeholder)

This pipeline includes:
- curate (min rating)
- train/eval split
- training stub (stats)
- evaluation stub

Apply:
  kubectl apply -f infra/learning/train-job.yaml
