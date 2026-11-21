# colormix

## 题面（可直接粘贴）

预乘 alpha 合成与混合模式。约束：必须在预乘空间做 source-over 合成（不得先混合再乘 alpha），结果与 W3C compositing 参考公式逐通道误差 ≤1e-12；必须实现 multiply、screen、overlay、soft-light 四种模式，且 overlay/soft-light 需按 W3C 分段公式（含 x≤0.5 与 x>0.5 两支）；alpha=1 时 multiply/screen 满足交换律；反预乘在 alpha=0 时返回 0 而不是 NaN；8 位量化往返误差 ≤1/255；非法模式、越界颜色、长度不匹配、非有限输入要给出可区分的错误；10 万像素不超过 300 ms；并发安全。验收：cd colormix && python -m pytest tests -q -k mix。改动限 2 到 3 个文件，增量构建加测试不超过 30 秒。不得改动既有测试与既有行为。交付给命令、输出、耗时三行自测证据。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd colormix && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
