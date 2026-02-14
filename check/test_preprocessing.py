"""
Test preprocessing.py
In ra output của các hàm
"""

import sys
sys.path.append('/kaggle/working/VLM-for-assisting-visual-impaired-human/src')

print("="*80)
print("🧪 TEST PREPROCESSING.PY")
print("="*80)

# ============================================================================
# Import
# ============================================================================
print("\n📦 Import preprocessing")
print("-"*80)

try:
    from data.preprocessing import *
    print("✅ Import thành công")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Liệt kê tất cả functions
# ============================================================================
print("\n📦 Các functions có trong preprocessing.py:")
print("-"*80)

import data.preprocessing as prep
import inspect

functions = [name for name, obj in inspect.getmembers(prep) 
             if inspect.isfunction(obj) and not name.startswith('_')]

for i, func_name in enumerate(functions, 1):
    func = getattr(prep, func_name)
    sig = inspect.signature(func)
    print(f"{i}. {func_name}{sig}")

# ============================================================================
# Test từng function với dummy data
# ============================================================================
print("\n" + "="*80)
print("🧪 TEST TỪNG FUNCTION")
print("="*80)

# Test từng function tìm thấy
for func_name in functions:
    print(f"\n📦 TEST: {func_name}()")
    print("-"*80)
    
    func = getattr(prep, func_name)
    
    try:
        # Lấy signature để tạo dummy args
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        print(f"Parameters: {params}")
        
        # Gọi function với dummy data tùy theo tên
        if 'image' in func_name.lower():
            # Tạo dummy image
            from PIL import Image
            import numpy as np
            dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            
            if len(params) == 1:
                result = func(dummy_img)
            elif len(params) == 2:
                result = func(dummy_img, (224, 224))
            else:
                print("  ⊘ Cần custom test cho function này")
                continue
                
        elif 'text' in func_name.lower() or 'prompt' in func_name.lower():
            # Dummy text
            dummy_text = "This is a test prompt with <image> placeholder"
            result = func(dummy_text)
            
        elif 'bbox' in func_name.lower():
            # Dummy bbox
            dummy_bbox = [100, 200, 300, 400]
            result = func(dummy_bbox)
            
        elif 'normalize' in func_name.lower():
            # Dummy tensor
            import torch
            dummy_tensor = torch.randn(3, 224, 224)
            result = func(dummy_tensor)
            
        else:
            print("  ⊘ Không biết cách test function này, skip")
            continue
        
        # In kết quả
        print(f"\n✅ Output:")
        print(f"  Type: {type(result)}")
        
        if isinstance(result, (list, tuple)):
            print(f"  Length: {len(result)}")
            if len(result) > 0:
                print(f"  First item type: {type(result[0])}")
                print(f"  First item: {result[0]}")
        elif hasattr(result, 'shape'):
            print(f"  Shape: {result.shape}")
            print(f"  Dtype: {result.dtype}")
        elif isinstance(result, dict):
            print(f"  Keys: {list(result.keys())}")
            for k, v in result.items():
                print(f"    {k}: {type(v)}")
        else:
            print(f"  Value: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ PREPROCESSING TEST COMPLETED")
print("="*80)
print(f"\nTổng số functions: {len(functions)}")
print("\n💡 Xem output ở trên để đánh giá!")