# hf-utils

`hf-utils`는 Huggingface Hub에 데이터를 업로드하고 관리하는 유틸리티입니다. 이 패키지는 `fire` 라이브러리를 사용하여 명령줄 인터페이스(CLI)를 제공합니다.

## 설치

```sh
pip install git+https://github.com/KU-HIAI/hf-utils
```

## 사용법

`hf-utils`는 명령줄에서 실행할 수 있습니다. 다음은 사용 가능한 명령어들입니다:

### hf-utils

#### upload_to_hub

데이터셋을 Huggingface Hub에 업로드합니다.

```sh
hf-utils upload_to_hub --repo_id <repo_id> [--is_private] --train_json <train_json_path> [--val_json <val_json_path>] [--test_json <test_json_path>] [--train_test_split_ratio <ratio>]
```

#### update_readme

현재 리포지토리의 README.md 파일을 업데이트합니다.

```sh
hf-utils update_readme --repo_id <repo_id>
```

#### update_all_datasets_readme

조직의 모든 데이터셋의 README.md 파일을 업데이트합니다.

```sh
hf-utils update_all_datasets_readme --organization <organization_name>
```

## 예제

#### 데이터셋 업로드

```sh
hf-utils upload_to_hub --repo_id my_org/my_dataset --is_private --train_json path/to/train.json --val_json path/to/val.json --test_json path/to/test.json --train_test_split_ratio 0.9
```

#### README.md 업데이트

```sh
hf-utils update_readme
```

#### 조직의 모든 데이터셋 README.md 업데이트


```sh
hf-utils update_all_datasets_readme --organization my_org
```