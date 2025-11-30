# Tensor Compiler

## 目標
- ONNX Runtimeのようなテンソルコンパイラを作成する
    - onnxファイルが入力されたら、適切なfuseを実行する
        - コスト関数を使用した最適なfuseを探索
        - islを使用したpolyhedral最適化
    - C/CUDAを出力
        - tile, unroll, vectorize, parallelize, simd, tensor core, cp.async, prefetchingなどを活用
    - コンパイラ最適化を実行
- Clang, NVRTCを使用してJITコンパイルを実行

## 実装計画
### ScheduleIRを用いた、ループ変換
1. Split
2. Reorder
3. Fuse
4. Unroll
5. Vectorize
