#!/usr/bin/env python3

import torch
import sys
sys.path.append('.')

def test_multi_object_expand_mask():
    """测试多对象expand mask的创建逻辑"""

    # 模拟参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_seq_len = 20
    visual_seq_len = 100
    id_len = 50  # 保存的key数量

    # 创建模拟的ID mask（哪些位置是前景）
    id_fg_mask = torch.zeros(id_len, dtype=torch.bool, device=device)
    id_fg_mask[:25] = True  # 前25个位置是前景

    # 单对象模式下的curr_fg_mask
    curr_fg_mask = torch.zeros(visual_seq_len, dtype=torch.bool, device=device)
    curr_fg_mask[:50] = True  # 前50个视觉位置是前景

    # 多对象模式下的curr_fg_masks
    curr_fg_masks = []
    # 对象1：前25个视觉位置
    obj1_mask = torch.zeros(visual_seq_len, dtype=torch.bool, device=device)
    obj1_mask[:25] = True
    curr_fg_masks.append(obj1_mask)

    # 对象2：25-50个视觉位置
    obj2_mask = torch.zeros(visual_seq_len, dtype=torch.bool, device=device)
    obj2_mask[25:50] = True
    curr_fg_masks.append(obj2_mask)

    # 创建模拟的processor
    class MockProcessor:
        def __init__(self):
            self.text_seq_len = text_seq_len
            self.visual_seq_len = visual_seq_len

        def get_expand_attn_mask(self, id_fg_mask, curr_fg_mask, bg_share_flag, fg_share_flag, device,
                               num_objects=1, curr_fg_masks=None):
            id_len = len(id_fg_mask)
            _id_fg_mask = id_fg_mask.view(1, 1, 1, -1)

            if num_objects == 1 or curr_fg_masks is None:
                # 单对象模式：保持原有逻辑
                _curr_fg_mask = curr_fg_mask.view(1, 1, -1, 1)
                expand_mask = _id_fg_mask != _curr_fg_mask
            else:
                # 多对象模式：为每个对象创建独立的mask，确保对象间不竞争
                expand_masks = []
                for obj_idx, obj_mask in enumerate(curr_fg_masks):
                    _obj_mask = obj_mask.view(1, 1, -1, 1)
                    obj_expand_mask = _id_fg_mask != _obj_mask

                    # 额外限制：当前对象的区域不能attend到其他对象的ID位置
                    for other_idx, other_mask in enumerate(curr_fg_masks):
                        if other_idx != obj_idx:
                            other_mask_flat = other_mask.view(1, 1, -1, 1)
                            # 当前对象的视觉位置不能attend到其他对象的ID位置
                            obj_expand_mask = obj_expand_mask | other_mask_flat

                    expand_masks.append(obj_expand_mask)

                # 合并所有对象的mask（使用OR操作确保任何对象都不被屏蔽）
                expand_mask = torch.stack(expand_masks, dim=0).any(dim=0)

            if not bg_share_flag:
                expand_mask[:, :, :, ~id_fg_mask] = True
            elif not fg_share_flag:
                expand_mask[:, :, :, id_fg_mask] = True
            t2i_expand_mask = torch.ones((1, 1, self.text_seq_len, id_len), device=device, dtype=bool)
            return torch.cat((t2i_expand_mask, expand_mask), dim=-2)

    processor = MockProcessor()

    print("=== Testing Multi-Object Expand Mask ===")

    # 测试单对象模式
    print("\n1. Testing single object mode...")
    single_mask = processor.get_expand_attn_mask(
        id_fg_mask, curr_fg_mask, bg_share_flag=True, fg_share_flag=True, device=device,
        num_objects=1, curr_fg_masks=None
    )
    print(f"Single object mask shape: {single_mask.shape}")
    print(f"Single object mask has {torch.sum(single_mask).item()} True values")

    # 测试多对象模式
    print("\n2. Testing multi-object mode...")
    multi_mask = processor.get_expand_attn_mask(
        id_fg_mask, curr_fg_mask, bg_share_flag=True, fg_share_flag=True, device=device,
        num_objects=2, curr_fg_masks=curr_fg_masks
    )
    print(f"Multi object mask shape: {multi_mask.shape}")
    print(f"Multi object mask has {torch.sum(multi_mask).item()} True values")

    # 验证多对象mask的逻辑
    print("\n3. Verifying multi-object mask logic...")

    # 检查对象1区域的行为 (视觉位置 10)
    visual_pos_obj1 = 10
    # 对象1的mask应该允许attend到ID的前景位置
    obj1_id_positions = torch.nonzero(id_fg_mask).flatten()

    print(f"Object 1 visual position {visual_pos_obj1} should attend to ID positions: {obj1_id_positions.tolist()}")

    # 检查对象1的视觉位置是否能attend到对象1的ID位置
    # 注意：mask的索引是 [batch, head, query_pos, key_pos]
    # 其中key_pos对应的是附加的ID位置（0到id_len-1）
    query_pos = text_seq_len + visual_pos_obj1

    for id_pos in obj1_id_positions[:5]:  # 只检查前5个
        mask_val = multi_mask[0, 0, query_pos, id_pos]
        if mask_val:
            print(f"✗ Object 1 visual pos {visual_pos_obj1} CANNOT attend to ID pos {id_pos.item()} (masked)")
        else:
            print(f"✓ Object 1 visual pos {visual_pos_obj1} CAN attend to ID pos {id_pos.item()}")

    # 比较单对象和多对象的差异
    print(f"\nMask difference (multi - single): {torch.sum(multi_mask != single_mask).item()} positions differ")
    print(f"Multi-object has {torch.sum(multi_mask).item()} masked positions")
    print(f"Single-object has {torch.sum(single_mask).item()} masked positions")

    print("\nTest completed!")

if __name__ == "__main__":
    test_multi_object_expand_mask()
