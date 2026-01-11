#!/usr/bin/env python3

import torch
import sys
sys.path.append('.')

def test_attention_mask():
    """测试attention mask的创建逻辑"""

    # 模拟参数
    batch_size = 1
    num_heads = 24
    text_len = 20
    visual_seq_len = 100
    bg_len = 3
    object_token_ranges = [(5, 10), (12, 17)]  # 对象1: tokens 5-9, 对象2: tokens 12-16

    # 创建position_to_object映射 (模拟两个对象的简单分布)
    position_to_object = torch.zeros(visual_seq_len, dtype=torch.long)
    position_to_object[:50] = 1  # 前半部分是对象1
    position_to_object[50:] = 2  # 后半部分是对象2

    # 创建attention mask
    total_seq_len = text_len + visual_seq_len
    attention_mask = torch.zeros((batch_size, num_heads, total_seq_len, total_seq_len), dtype=torch.float32)

    print("=== Testing Attention Mask Creation ===")
    print(f"Total sequence length: {total_seq_len}")
    print(f"Text length: {text_len}, Visual length: {visual_seq_len}")
    print(f"Background tokens: 0-{bg_len-1}")
    print(f"Object 1 tokens: {object_token_ranges[0]}")
    print(f"Object 2 tokens: {object_token_ranges[1]}")

    # 1. 设置文本内部attention限制
    print("\n1. Setting text-to-text attention restrictions...")
    for text_pos in range(text_len):
        # 确定这个文本token属于哪个对象
        token_obj_id = 0  # 背景
        if text_pos >= bg_len:
            for obj_idx, (start_token, end_token) in enumerate(object_token_ranges):
                if start_token <= text_pos < end_token:
                    token_obj_id = obj_idx + 1
                    break

        # 这个文本token只能attend到相同对象的其他tokens
        for other_text_pos in range(text_len):
            other_token_obj_id = 0  # 背景
            if other_text_pos >= bg_len:
                for obj_idx, (start_token, end_token) in enumerate(object_token_ranges):
                    if start_token <= other_text_pos < end_token:
                        other_token_obj_id = obj_idx + 1
                        break

            # 如果是不同对象的tokens，屏蔽attention
            if token_obj_id != other_token_obj_id and token_obj_id != 0 and other_token_obj_id != 0:
                attention_mask[0, :, text_pos, other_text_pos] = float("-inf")

    # 2. 设置视觉到文本的attention限制
    print("2. Setting visual-to-text attention restrictions...")
    for visual_pos in range(visual_seq_len):
        obj_id = position_to_object[visual_pos]

        # 确定这个视觉位置可以attend的文本tokens
        allowed_text_tokens = torch.zeros(text_len, dtype=torch.bool)

        # 所有位置都可以看到背景tokens
        allowed_text_tokens[:bg_len] = True

        # 对象位置还可以看到对应对象的tokens
        if obj_id > 0 and obj_id <= len(object_token_ranges):
            start_token, end_token = object_token_ranges[obj_id - 1]
            allowed_text_tokens[start_token:end_token] = True

        # 设置attention mask：视觉位置只能attend到允许的文本tokens
        query_pos_in_total = text_len + visual_pos

        for text_token_idx in range(text_len):
            if not allowed_text_tokens[text_token_idx]:
                attention_mask[0, :, query_pos_in_total, text_token_idx] = float("-inf")

    # 验证结果
    print("\n3. Verifying attention mask...")

    # 检查对象1区域的视觉位置 (position 25, 对象1)
    visual_pos_obj1 = 25
    query_pos_obj1 = text_len + visual_pos_obj1

    obj1_allowed = torch.zeros(text_len, dtype=torch.bool)
    obj1_allowed[:bg_len] = True
    start_token, end_token = object_token_ranges[0]
    obj1_allowed[start_token:end_token] = True

    print(f"Object 1 visual position {visual_pos_obj1} should attend to tokens: {torch.nonzero(obj1_allowed).flatten().tolist()}")

    # 检查mask是否正确设置
    masked_positions = 0
    for text_token_idx in range(text_len):
        if attention_mask[0, 0, query_pos_obj1, text_token_idx] == float("-inf"):
            if obj1_allowed[text_token_idx]:
                print(f"ERROR: Object 1 should attend to token {text_token_idx} but it's masked!")
            else:
                masked_positions += 1

    print(f"Correctly masked {masked_positions} positions for object 1")

    # 检查对象2区域的视觉位置 (position 75, 对象2)
    visual_pos_obj2 = 75
    query_pos_obj2 = text_len + visual_pos_obj2

    obj2_allowed = torch.zeros(text_len, dtype=torch.bool)
    obj2_allowed[:bg_len] = True
    start_token, end_token = object_token_ranges[1]
    obj2_allowed[start_token:end_token] = True

    print(f"Object 2 visual position {visual_pos_obj2} should attend to tokens: {torch.nonzero(obj2_allowed).flatten().tolist()}")

    # 检查对象2的mask
    masked_positions = 0
    for text_token_idx in range(text_len):
        if attention_mask[0, 0, query_pos_obj2, text_token_idx] == float("-inf"):
            if obj2_allowed[text_token_idx]:
                print(f"ERROR: Object 2 should attend to token {text_token_idx} but it's masked!")
            else:
                masked_positions += 1

    print(f"Correctly masked {masked_positions} positions for object 2")

    # 检查文本内部attention
    print("\n4. Checking text-to-text attention...")

    # 对象1的token (比如token 7) 应该不能attend到对象2的token (比如token 14)
    obj1_token = 7
    obj2_token = 14

    if attention_mask[0, 0, obj1_token, obj2_token] == float("-inf"):
        print(f"✓ Object 1 token {obj1_token} correctly cannot attend to object 2 token {obj2_token}")
    else:
        print(f"✗ ERROR: Object 1 token {obj1_token} can attend to object 2 token {obj2_token}")

    # 对象1的token应该能attend到背景token
    bg_token = 2
    if attention_mask[0, 0, obj1_token, bg_token] != float("-inf"):
        print(f"✓ Object 1 token {obj1_token} can attend to background token {bg_token}")
    else:
        print(f"✗ ERROR: Object 1 token {obj1_token} cannot attend to background token {bg_token}")

    total_masked = torch.sum(attention_mask == float("-inf")).item()
    print(f"\nTotal masked positions: {total_masked} out of {attention_mask.numel()}")
    print("Test completed!")

if __name__ == "__main__":
    test_attention_mask()
