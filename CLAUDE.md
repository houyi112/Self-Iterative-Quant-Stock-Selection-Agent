# CLAUDE.md

## 自动执行（无需确认）

### 所有 Bash 命令
- 所有 `Bash` 命令默认自动执行
- 包括：curl、python3、cat、ls、grep、awk、jq、sort、head、tail、iconv、echo、for、while 等

### 所有搜索和抓取
- `WebSearch` 所有搜索请求
- `WebFetch` 所有网页抓取

### 所有 Agent
- `Agent` 启动子 Agent 执行任务

### 文件读写（安全路径）
- `Write` / `Edit` 记忆文件：`/Users/bytedance/.claude/projects/-Users-bytedance-claude-code/memory/`
- `Write` / `Edit` 当前项目下 `.md`、`.json`、`.txt` 文件

---

## 必须确认（每次弹框）

### 删除
- rm、rmdir、git rm
- 任何文件删除

### Git 高危
- git push、git push --force
- git reset --hard、git clean
- git rebase、git commit --amend
- 修改 .gitconfig

### 系统
- sudo
- chmod、chown
- brew install、npm install -g
- 修改 .bashrc、.zshrc、.profile

### 安全
- 向外部服务发送数据
- 操作账户密码、Token、密钥
