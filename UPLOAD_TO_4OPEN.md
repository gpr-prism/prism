# 上传到 anonymous.4open.science 步骤

anonymous.4open.science 需要先有 **GitHub 公开仓库**，再通过其页面生成匿名链接。

## 第一步：在 GitHub 创建仓库

1. 打开 https://github.com/new
2. **Repository name** 填：`prism-graph-learning`（或任意名称）
3. 选择 **Public**
4. **不要**勾选 "Add a README file"（本地已有）
5. 点击 **Create repository**

## 第二步：推送代码到 GitHub

在终端执行（将 `YOUR_USERNAME` 替换为你的 GitHub 用户名）：

```bash
cd /Users/xuxin/Desktop/prism

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/prism-graph-learning.git

# 推送
git push -u origin main
```

若使用 SSH：
```bash
git remote add origin git@github.com:YOUR_USERNAME/prism-graph-learning.git
git push -u origin main
```

## 第三步：在 4open.science 匿名化

1. 打开：https://anonymous.4open.science/anonymize/msir-F607  
2. 用 GitHub 账号登录  
3. 输入你的仓库 URL，例如：`https://github.com/YOUR_USERNAME/prism-graph-learning`  
4. 按提示完成匿名化，获得匿名链接提交给审稿人  

---

**说明：** `llm_model/`（约 421MB）已加入 `.gitignore`，因 GitHub 单文件限制 100MB。如需提供模型，可上传到 Hugging Face 或在 README 中注明下载方式。
