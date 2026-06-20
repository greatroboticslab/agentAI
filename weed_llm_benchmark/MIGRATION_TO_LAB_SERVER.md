# 迁移到 Lab Ubuntu Server (Unie) — 方案 + 步骤

教授要求:cluster 只剩几个月、且 cluster 是给计算用的不适合做 labeling/web。
所以把 **dashboard + MongoDB + 数据集图片库** 迁到 lab 自己的 Ubuntu server,
cluster 只保留 **采集(harvest)+ 训练(GPU)**。Roboflow 不付费,只做临时人工标注
(本地上传 → 标 → 拉回 → 清除)。

## 目标架构

```
┌─────────────────────── Lab Ubuntu Server (Unie, 常开, 无需 GPU) ───────────────────────┐
│  • MongoDB            ← 真源 (source of truth) + 标注生命周期/历史                        │
│  • FastAPI dashboard  ← web server (systemd 常驻), 固定地址, 不再用 cloudflare 临时隧道   │
│  • 数据集图片库 (本地 SSD) ← /classes /slugs 等可视化读本地盘 = 秒开 (根治 Lustre 超时)   │
│  • 临时 Roboflow 桥   ← 需要人工标时上传几张 → 标完拉回 → 清 Roboflow (省钱)             │
└────────────────────────────────────────────────────────────────────────────────────────┘
                ▲  (lab 主动 pull, 见下方"同步方向")
                │  rsync over ssh  +  mongo 记录导入
┌───────────────┴─────────── Bridges-2 Cluster (GPU, 几个月内) ────────────────────────────┐
│  • 采集 harvest jobs   → 写 cluster 本地 (registry/JSON/本地 mongo 或直接产出)            │
│  • 训练 jobs (RF-DETR/YOLO) → 读数据训练                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 关键设计决定:同步方向 = **lab 主动从 cluster 拉**(不是 cluster 推 lab)

Bridges-2 计算节点对外网/校园内网的出站连接通常受限,让 cluster 主动连 lab server
(写 Mongo / 推图)往往被防火墙挡。反过来 **lab server → cluster (ssh/rsync 出站)** 简单可靠。
所以:lab server 上跑一个定时同步(cron/systemd timer),周期性地:
1. `rsync` 把 cluster 上新采集的图片拉到 lab 本地 SSD;
2. 导入 cluster 新产出的 harvest 记录到 lab 的 MongoDB(或直接读取 dual-write 的 JSON 回放)。
训练需要的"人工已标 ground truth"则由 lab → cluster 推过去(出站,简单)。

## 迁移步骤(我需要 SSH 到 lab server 后执行)

### 0. 给我 SSH 访问(最关键前置 — 见下)
### 1. lab server 装环境(`deploy/lab_server_setup.sh`)
- MongoDB Community、Python venv、克隆 repo、装依赖(fastapi/uvicorn/ultralytics 等可选)
- dashboard 设为 **systemd service**(常驻、开机自启、崩溃自重启)—— 不再靠 sbatch/隧道
### 2. 迁 MongoDB
- cluster: `mongodump`(当前库)→ scp/rsync 到 lab → `mongorestore`
### 3. 迁数据集图片库
- `rsync -az` cluster `/ocean/.../downloads` + `results/framework` 关键产物 → lab 本地 SSD
### 4. 重新指向
- dashboard `REPO_ROOT`/Mongo URL 指向 lab 本地;`/classes` 读本地盘
- 暴露访问:校内固定地址 / 或 Tailscale 地址 / 或一个稳定隧道(不再每次轮换 URL)
### 5. 持续同步(`deploy/sync_from_cluster.sh` + systemd timer)
- 周期 pull 新采集数据 + 导入记录;labeled ground truth 反向推 cluster 供训练
### 6. 备份(lab 单机=单点故障)
- 定时 `mongodump` + 图片库快照到第二块盘/网盘

## 我需要你提供的(Fri-Sun 你方便时)
1. **SSH 访问 lab server**(见下,强烈建议 Tailscale)
2. 服务器信息:Ubuntu 版本、磁盘可用空间、有没有 sudo、有没有 GPU(dashboard 不需要)
3. 确认:Mongo 装在 lab server 本机(我假设是)

## 当前已就绪(本周已建+验证,迁过去即用)
- 采集 agent、DINOv2 采集期过滤(已验证拦截 coconut/beehive)、push-guard
- 人工标注闭环 A/B/C:push→agent标→human标→verify→delete→repush(已端到端验证 0→3)
- MongoDB 生命周期 + 历史、真实按钮状态、Roboflow allow-list
- 代码本就可移植(REPO_ROOT env、Mongo 跨节点、FastAPI)→ 迁移=搬数据+配网,非重写
