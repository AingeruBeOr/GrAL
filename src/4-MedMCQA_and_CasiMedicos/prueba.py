from dataset import MedMCQAAndCasiMedicosDataset, CasiMedicosDatasetBalanced

train_path = "../../data/casiMedicos-balanced/JSONL/en.train_casimedicos.jsonl"
dev_path = "../../data/casiMedicos-balanced/JSONL/en.dev_casimedicos.jsonl"
test_path = "../../data/casiMedicos-balanced/JSONL/en.test_casimedicos.jsonl"
use_context = False

train_dataset = MedMCQAAndCasiMedicosDataset(
        casimedicos_dataset_path=train_path, 
        medmcqa_dataset_path='../../data/MedMCQA-balanced/train.json',
        use_context=use_context
    )
test_dataset = CasiMedicosDatasetBalanced(test_path, use_context)
val_dataset = CasiMedicosDatasetBalanced(dev_path, use_context)

print(f"Train dataset: {len(train_dataset)}")
print(f"Test dataset: {len(test_dataset)}")
print(f"Val dataset: {len(val_dataset)}")