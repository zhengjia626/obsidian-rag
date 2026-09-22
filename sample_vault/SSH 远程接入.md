---
tags: [network, ssh, windows]
---

# SSH 远程接入

在公司 118 与家庭 112/105 之间，通过 SSH 协作访问多台机器。

## Windows 开启 OpenSSH

Windows 11 自带 OpenSSH 服务端，需在"可选功能"中启用，并启动 `sshd` 服务。

## 公钥 + 密码双认证

编辑 `C:\ProgramData\ssh\sshd_config`，确保 `PubkeyAuthentication yes` 与
`PasswordAuthentication yes` 同时开启，实现"密钥便捷 + 密码兜底"。

## 多机协作

家庭网络内 112 笔记本、105 Mac mini，公司 118 笔记本，均通过 Tailscale 组网后
再用 SSH 互访，避免暴露 22 端口到公网。
