# Mac 上将项目上传到 GitHub 的简明教程

以下命令适用于 macOS、Python3 和已安装 Git 的环境。

## 1. 进入项目目录

```bash
cd a-share-financial-distress-risk-prediction
```

## 2. 本地测试代码

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

python3 src/financial_risk_pipeline.py   --input data/example/synthetic_financial_data.csv   --output_dir outputs/demo   --targets Oscore Zrisk   --sample_train_rows 1000   --rf_n_estimators 30   --lgbm_n_estimators 50   --n_jobs 1   --nn_epochs 5   --permutation_repeats 1   --no_permutation
```

## 3. 清理 Mac 临时文件

```bash
find . -name ".DS_Store" -delete
find . -name "__MACOSX" -type d -exec rm -rf {} +
find . -name "._*" -delete
```

## 4. 初始化 Git 并提交

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial commit: financial distress risk prediction pipeline"
```

在 `git status` 中确认没有真实原始数据、Word 报告、模型文件、压缩包或个人信息。

## 5. 在 GitHub 创建仓库

建议仓库名：

```text
a-share-financial-distress-risk-prediction
```

仓库简介：

```text
A leakage-free machine-learning pipeline for A-share listed company financial distress risk prediction with Oscore and Zrisk robustness checks.
```

创建仓库时不要勾选 README、.gitignore 或 License，因为本项目已经包含这些文件。

## 6. 连接远程仓库并推送

把 `<your-username>` 替换成你的 GitHub 用户名：

```bash
git remote add origin https://github.com/<your-username>/a-share-financial-distress-risk-prediction.git
git push -u origin main
```

如果 GitHub 要求输入密码，密码位置应粘贴 GitHub Personal Access Token，而不是账户登录密码。

## 7. 上传后检查

打开 GitHub 仓库，确认：

- README 能正常显示；
- `src/financial_risk_pipeline.py` 存在；
- `reports/financial_distress_research_report_zh.md` 能正常打开；
- 没有上传真实原始数据、模型文件、个人身份信息和 Word 报告。
