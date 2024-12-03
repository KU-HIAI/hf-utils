# python libraries
import os
import json
import yaml
import tempfile
import traceback
# 3rd-party libraries
import fire
import structlog
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# setup logger
logger = structlog.get_logger()

class HFUploader:
    def __init__(self, repo_id=None, is_private=True):
        self.repo_id = repo_id
        self.is_private = is_private
        self.api = HfApi()

    def dataset_upload(self, train_json, val_json=None, test_json=None, train_test_split_ratio=0):
        with open(train_json) as fp:
            train_data = [json.loads(line) for line in fp]

        dataset_dict = {'train': Dataset.from_list(train_data)}

        if val_json:
            with open(val_json) as fp:
                val_data = [json.loads(line) for line in fp]
            dataset_dict['validation'] = Dataset.from_list(val_data)
        elif train_test_split_ratio > 0:
            train_data, val_data = train_test_split(train_data, test_size=1-train_test_split_ratio, random_state=42)
            dataset_dict = {
                'train': Dataset.from_list(train_data),
                'validation': Dataset.from_list(val_data)
            }

        if test_json:
            with open(test_json) as fp:
                test_data = [json.loads(line) for line in fp]
            dataset_dict['test'] = Dataset.from_list(test_data)

        dataset = DatasetDict(dataset_dict)

        self.api.create_repo(self.repo_id, private=self.is_private, repo_type='dataset', exist_ok=True)
        dataset.push_to_hub(self.repo_id, commit_message='Initial commit')
        
        # 데이터셋 업로드 후 README 업데이트
        self.update_dataset_readme()

    def update_dataset_readme(self):
        # repo_id 출력
        logger.info("Updating README", repo_id=self.repo_id)
        
        # 임시 폴더에서 작업
        with tempfile.TemporaryDirectory() as tmpdirname:
            # README.md 파일 다운로드
            downloaded_readme_path = hf_hub_download(repo_id=self.repo_id, filename='README.md', repo_type='dataset', local_dir=tmpdirname)
            with open(downloaded_readme_path, 'r') as f:
                readme_content = f.read()
                
            if '---' not in readme_content:
                logger.info("No YAML metadata found. Skipping update.")
                return

            README = readme_content.split('---')[2]
            if len(README) > 10:
                logger.info("Already has README.md contents. Skipping update.")
                return

            # YAML 메타데이터 파싱
            yaml_content = readme_content.split('---')[1]
            metadata = yaml.safe_load(yaml_content)

            # 데이터 개수와 크기 정보 추출
            splits_info = metadata['dataset_info']['splits']
            split_stats = {}
            for split in splits_info:
                split_name = split['name']
                split_stats[split_name] = {
                    'num_examples': int(split['num_examples']),
                    'num_bytes': int(split['num_bytes'])
                }

            # 샘플 데이터 생성
            dataset = load_dataset(self.repo_id)
            sample_data = {}
            for split_name in split_stats.keys():
                sample_data[f'{split_name}_sample'] = dataset[split_name].shuffle(seed=42).select(range(5))

            # 표 형식으로 샘플 데이터를 생성하는 함수
            def generate_table(data):
                headers = data.column_names
                table = "| " + " | ".join(headers) + " |\n"
                table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in data:
                    table += "| " + " | ".join(str(row[col]).replace('\n', '</br>') for col in headers) + " |\n"
                return table

            # model_card 내용 작성
            model_card_content = f"""
# Dataset Card for {self.repo_id}

## Dataset Statistics

| Split | # Examples | Size (bytes) |
|-------|------------|--------------|
"""
            for split_name, stats in split_stats.items():
                model_card_content += f"| {split_name.capitalize()} | {stats['num_examples']:,} | {int(stats['num_bytes']):,} |\n"

            model_card_content += """
## Dataset Structure
"""
            for split_name, stats in split_stats.items():
                model_card_content += f"""
### {split_name.capitalize()} Sample
{generate_table(sample_data[f'{split_name}_sample'])}
"""

            # model_card 파일 생성
            readme_path = os.path.join(tmpdirname, 'README.md')
            with open(readme_path, 'w') as f:
                f.write(f"---\n{yaml_content}\n---\n" + model_card_content)

            self.api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo='README.md',
                repo_id=self.repo_id,
                repo_type='dataset'
            )

    def update_all_datasets_readme(self, organization):
        datasets = self.api.list_datasets(author=organization)
        for dataset in datasets:
            try:
                self.repo_id = dataset.id
                self.update_readme()
            except Exception as ex:
                logger.error("Error updating README", repo_id=self.repo_id, error=ex)
                logger.error(traceback.print_exc())

def main():
    fire.Fire(HFUploader)

if __name__ == '__main__':
    main()