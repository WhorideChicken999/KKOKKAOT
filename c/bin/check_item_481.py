import psycopg2

# DB 연결
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='kkokkaot_closet',
    user='postgres',
    password='000000'
)

try:
    with conn.cursor() as cur:
        print("\n" + "="*80)
        print("item_id 481 check")
        print("="*80 + "\n")
        
        # 1. wardrobe_items 확인
        cur.execute("""
            SELECT item_id, has_outer, has_top, has_bottom, has_dress
            FROM wardrobe_items 
            WHERE item_id = 481
        """)
        
        wardrobe = cur.fetchone()
        if wardrobe:
            print("wardrobe_items:")
            print(f"  - item_id: {wardrobe[0]}")
            print(f"  - has_outer: {wardrobe[1]}")
            print(f"  - has_top: {wardrobe[2]}")
            print(f"  - has_bottom: {wardrobe[3]}")
            print(f"  - has_dress: {wardrobe[4]}")
        else:
            print("ERROR: wardrobe_items not found!")
            exit()
        
        # 2. outer_attributes 확인
        print("\n" + "-"*80)
        cur.execute("""
            SELECT outer_id, item_id, category, color, fit, materials, 
                   category_confidence, color_confidence, fit_confidence
            FROM outer_attributes 
            WHERE item_id = 481
        """)
        
        outer = cur.fetchone()
        if outer:
            print("outer_attributes:")
            print(f"  - outer_id: {outer[0]}")
            print(f"  - item_id: {outer[1]}")
            print(f"  - category: {outer[2]}")
            print(f"  - color: {outer[3]}")
            print(f"  - fit: {outer[4]}")
            print(f"  - materials: {outer[5]}")
            print(f"  - category_confidence: {outer[6]}")
            print(f"  - color_confidence: {outer[7]}")
            print(f"  - fit_confidence: {outer[8]}")
        else:
            print("ERROR: outer_attributes not found!")
            print("   => has_outer=True but outer_attributes table is empty")
        
        # 3. top_attributes 확인 (혹시 여기에 저장되었나?)
        print("\n" + "-"*80)
        cur.execute("""
            SELECT top_id, item_id, category, color, fit, materials
            FROM top_attributes 
            WHERE item_id = 481
        """)
        
        top = cur.fetchone()
        if top:
            print("top_attributes (maybe saved here?):")
            print(f"  - top_id: {top[0]}")
            print(f"  - item_id: {top[1]}")
            print(f"  - category: {top[2]}")
            print(f"  - color: {top[3]}")
            print(f"  - fit: {top[4]}")
            print(f"  - materials: {top[5]}")
        else:
            print("WARNING: top_attributes also not found")
        
        print("\n" + "="*80 + "\n")
        
finally:
    conn.close()

