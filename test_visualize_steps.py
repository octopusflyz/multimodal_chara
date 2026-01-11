#!/usr/bin/env python3

def test_visualize_steps_logic():
    """测试可视化步骤的逻辑"""

    # 模拟参数
    visualize_denoise_steps = [10, 20, 30]
    num_inference_steps = 50

    print("=== Testing Visualize Steps Logic ===")
    print(f"Target steps: {visualize_denoise_steps}")
    print(f"Total steps: {num_inference_steps}")

    # 模拟去噪循环
    for i in range(num_inference_steps):
        # 检查是否需要可视化此步骤
        save_denoise_mask = (visualize_denoise_steps is not None and
                           (i + 1) in visualize_denoise_steps)

        # 模拟get_consist_kwargs的结果（正常情况下save_attn_weight=False）
        save_attn_weight = False  # 假设正常步骤不计算mask

        print(f"Step {i+1}: save_denoise_mask={save_denoise_mask}, save_attn_weight={save_attn_weight}")

        # 如果需要可视化，启用mask计算
        if save_denoise_mask and not save_attn_weight:
            save_attn_weight = True
            print(f"Step {i+1}: ENABLED mask calculation for visualization")

        # 模拟mask计算和可视化保存
        if save_attn_weight:
            print(f"Step {i+1}: Computing masks...")
            if save_denoise_mask:
                print(f"Step {i+1}: SAVING visualization!")

        # 只显示相关的步骤
        if (i + 1) in visualize_denoise_steps or i < 3:
            continue
        elif i >= 5:
            break

    print("\nExpected behavior:")
    print("- Step 10: Should enable mask calculation and save visualization")
    print("- Step 20: Should enable mask calculation and save visualization")
    print("- Step 30: Should enable mask calculation and save visualization")
    print("- Other steps: Should NOT enable mask calculation")

if __name__ == "__main__":
    test_visualize_steps_logic()
