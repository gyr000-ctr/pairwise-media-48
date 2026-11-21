# tilecollide

## 题面（可直接粘贴）

瓦片碰撞解算。约束：必须做 swept AABB（高速移动不穿透）；分离方向取最小位移轴且多次迭代后稳定；多瓦片重叠结果与逐瓦片检测一致；10 万次不超过 300 ms；非法尺寸可区分报错；并发安全。验收：cd tilecollide && python -m pytest tests -q -k collide。改动限 2 到 3 个文件，增量构建加测试不超过 30 秒。不得改动既有测试与既有行为。交付给命令、输出、耗时三行自测证据。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd tilecollide && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
