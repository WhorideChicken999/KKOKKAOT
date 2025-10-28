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
        print("Checking for duplicate outer_attributes entries")
        print("="*80 + "\n")
        
        # 같은 item_id가 outer_attributes에 여러 번 있는지 확인
        cur.execute("""
            SELECT item_id, COUNT(*) as count
            FROM outer_attributes
            GROUP BY item_id
            HAVING COUNT(*) > 1
            ORDER BY item_id
        """)
        
        duplicates = cur.fetchall()
        if duplicates:
            print(f"ERROR: Found {len(duplicates)} items with duplicate outer_attributes:")
            for dup in duplicates:
                print(f"  - item_id={dup[0]}: {dup[1]} rows")
                
                # 상세 정보 확인
                cur.execute("""
                    SELECT outer_id, category, color, fit, materials
                    FROM outer_attributes
                    WHERE item_id = %s
                """, (dup[0],))
                
                rows = cur.fetchall()
                for row in rows:
                    print(f"      outer_id={row[0]}: {row[1]}, {row[2]}, {row[3]}, {row[4]}")
        else:
            print("OK: No duplicate outer_attributes found")
        
        # user_id=3의 아우터 아이템 확인
        print("\n" + "-"*80)
        print("User 3's outer items:")
        print("-"*80)
        
        cur.execute("""
            SELECT w.item_id, o.category, o.color, o.fit, o.category_confidence
            FROM wardrobe_items w
            INNER JOIN outer_attributes o ON w.item_id = o.item_id
            WHERE w.user_id = 3 AND w.has_outer = TRUE
            ORDER BY w.item_id DESC
            LIMIT 5
        """)
        
        items = cur.fetchall()
        if items:
            print(f"Found {len(items)} outer items:")
            for item in items:
                conf_pct = int(item[4] * 100) if item[4] else 0
                print(f"  item_id={item[0]}: {item[1]} ({conf_pct}%), {item[2]}, {item[3]}")
        else:
            print("No outer items found for user 3")
        
        print("\n" + "="*80 + "\n")
        
finally:
    conn.close()

