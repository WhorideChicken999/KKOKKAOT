import psycopg2
from psycopg2.extras import Json

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
        print("Checking ALL items for outer/dress issues")
        print("="*80 + "\n")
        
        # 1. has_outer=True인데 outer_attributes가 없는 아이템
        print("[1] Checking OUTER items...")
        print("-"*80)
        cur.execute("""
            SELECT w.item_id, w.has_outer, w.has_top, w.has_bottom, w.has_dress
            FROM wardrobe_items w
            LEFT JOIN outer_attributes o ON w.item_id = o.item_id
            WHERE w.has_outer = TRUE AND o.outer_id IS NULL
            ORDER BY w.item_id
        """)
        
        missing_outer = cur.fetchall()
        if missing_outer:
            print(f"ERROR: Found {len(missing_outer)} items with has_outer=TRUE but NO outer_attributes:")
            for item in missing_outer:
                print(f"  - item_id={item[0]}: has_outer={item[1]}, has_top={item[2]}, has_bottom={item[3]}, has_dress={item[4]}")
        else:
            print("OK: All outer items have outer_attributes data")
        
        # 2. has_dress=True인데 dress_attributes가 없는 아이템
        print("\n[2] Checking DRESS items...")
        print("-"*80)
        cur.execute("""
            SELECT w.item_id, w.has_dress, w.has_top, w.has_bottom, w.has_outer
            FROM wardrobe_items w
            LEFT JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = TRUE AND d.dress_id IS NULL
            ORDER BY w.item_id
        """)
        
        missing_dress = cur.fetchall()
        if missing_dress:
            print(f"ERROR: Found {len(missing_dress)} items with has_dress=TRUE but NO dress_attributes:")
            for item in missing_dress:
                print(f"  - item_id={item[0]}: has_dress={item[1]}, has_top={item[2]}, has_bottom={item[3]}, has_outer={item[4]}")
        else:
            print("OK: All dress items have dress_attributes data")
        
        # 3. has_outer=False인데 outer_attributes가 있는 아이템 (반대 케이스)
        print("\n[3] Checking reverse issue (has_outer=FALSE but outer_attributes EXISTS)...")
        print("-"*80)
        cur.execute("""
            SELECT w.item_id, w.has_outer, o.category, o.color
            FROM wardrobe_items w
            INNER JOIN outer_attributes o ON w.item_id = o.item_id
            WHERE w.has_outer = FALSE
            ORDER BY w.item_id
        """)
        
        reverse_outer = cur.fetchall()
        if reverse_outer:
            print(f"WARNING: Found {len(reverse_outer)} items with has_outer=FALSE but outer_attributes EXISTS:")
            for item in reverse_outer:
                print(f"  - item_id={item[0]}: has_outer={item[1]}, category={item[2]}, color={item[3]}")
        else:
            print("OK: No reverse issues found")
        
        # 4. has_dress=False인데 dress_attributes가 있는 아이템
        print("\n[4] Checking reverse issue (has_dress=FALSE but dress_attributes EXISTS)...")
        print("-"*80)
        cur.execute("""
            SELECT w.item_id, w.has_dress, d.category, d.color
            FROM wardrobe_items w
            INNER JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = FALSE
            ORDER BY w.item_id
        """)
        
        reverse_dress = cur.fetchall()
        if reverse_dress:
            print(f"WARNING: Found {len(reverse_dress)} items with has_dress=FALSE but dress_attributes EXISTS:")
            for item in reverse_dress:
                print(f"  - item_id={item[0]}: has_dress={item[1]}, category={item[2]}, color={item[3]}")
        else:
            print("OK: No reverse issues found")
        
        # 5. 통계 요약
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        
        # 전체 아이템 수
        cur.execute("SELECT COUNT(*) FROM wardrobe_items")
        total = cur.fetchone()[0]
        print(f"Total items: {total}")
        
        # 카테고리별 개수
        cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE has_outer = TRUE")
        outer_count = cur.fetchone()[0]
        print(f"has_outer=TRUE: {outer_count}")
        
        cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE has_dress = TRUE")
        dress_count = cur.fetchone()[0]
        print(f"has_dress=TRUE: {dress_count}")
        
        cur.execute("SELECT COUNT(*) FROM outer_attributes")
        outer_attrs_count = cur.fetchone()[0]
        print(f"outer_attributes rows: {outer_attrs_count}")
        
        cur.execute("SELECT COUNT(*) FROM dress_attributes")
        dress_attrs_count = cur.fetchone()[0]
        print(f"dress_attributes rows: {dress_attrs_count}")
        
        print("\n" + "="*80)
        
        # 6. 문제가 있는 아이템이 있으면 수정 여부 물어보기
        if missing_outer or missing_dress:
            print("\nISSUES FOUND!")
            print(f"  - Missing outer_attributes: {len(missing_outer)}")
            print(f"  - Missing dress_attributes: {len(missing_dress)}")
            print("\nTo fix these issues, check if they have data in top_attributes")
        else:
            print("\nALL GOOD! No issues found.")
        
        print("="*80 + "\n")
        
finally:
    conn.close()

