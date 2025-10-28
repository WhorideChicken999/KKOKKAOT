import psycopg2

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
        print("Fixing item_id 483")
        print("="*80 + "\n")
        
        # 1. 현재 상태 확인
        cur.execute("""
            SELECT has_top, has_outer, has_bottom, has_dress
            FROM wardrobe_items
            WHERE item_id = 483
        """)
        flags = cur.fetchone()
        print(f"Before: has_top={flags[0]}, has_outer={flags[1]}, has_bottom={flags[2]}, has_dress={flags[3]}")
        
        # 2. has_top을 FALSE로 변경
        cur.execute("""
            UPDATE wardrobe_items
            SET has_top = FALSE
            WHERE item_id = 483
        """)
        print("=> Updated has_top to FALSE")
        
        # 3. top_attributes 삭제 (있다면)
        cur.execute("""
            SELECT COUNT(*) FROM top_attributes WHERE item_id = 483
        """)
        top_count = cur.fetchone()[0]
        
        if top_count > 0:
            cur.execute("DELETE FROM top_attributes WHERE item_id = 483")
            print(f"=> Deleted {top_count} row(s) from top_attributes")
        else:
            print("=> No top_attributes to delete")
        
        # 4. COMMIT
        conn.commit()
        print("\n=> SUCCESS!")
        
        # 5. 검증
        print("\n" + "-"*80)
        print("Verification:")
        print("-"*80)
        
        cur.execute("""
            SELECT has_top, has_outer
            FROM wardrobe_items
            WHERE item_id = 483
        """)
        flags = cur.fetchone()
        print(f"After: has_top={flags[0]}, has_outer={flags[1]}")
        
        cur.execute("""
            SELECT category, color, fit
            FROM outer_attributes
            WHERE item_id = 483
        """)
        outer = cur.fetchone()
        if outer:
            print(f"outer_attributes: {outer[0]}, {outer[1]}, {outer[2]}")
        
        print("\n" + "="*80 + "\n")
        
finally:
    conn.close()

