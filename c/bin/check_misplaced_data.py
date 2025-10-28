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
        print("Checking if outer/dress data is in top_attributes table")
        print("="*80 + "\n")
        
        # has_outer=True인 아이템이 top_attributes에 있는지 확인
        print("[1] has_outer=TRUE items in top_attributes:")
        print("-"*80)
        cur.execute("""
            SELECT w.item_id, w.has_outer, w.has_top, t.category, t.color, t.fit
            FROM wardrobe_items w
            INNER JOIN top_attributes t ON w.item_id = t.item_id
            LEFT JOIN outer_attributes o ON w.item_id = o.item_id
            WHERE w.has_outer = TRUE AND o.outer_id IS NULL
            ORDER BY w.item_id
            LIMIT 10
        """)
        
        rows = cur.fetchall()
        if rows:
            print(f"Found {len(rows)} items (showing first 10):")
            for row in rows:
                print(f"  item_id={row[0]}: has_outer={row[1]}, has_top={row[2]}, top_attrs=({row[3]}, {row[4]}, {row[5]})")
            print("\nCONFIRMED: outer data is in top_attributes!")
        else:
            print("No data found in top_attributes")
        
        # has_dress=True인 아이템이 top_attributes에 있는지 확인
        print("\n[2] has_dress=TRUE items in top_attributes:")
        print("-"*80)
        cur.execute("""
            SELECT w.item_id, w.has_dress, w.has_top, t.category, t.color, t.fit
            FROM wardrobe_items w
            INNER JOIN top_attributes t ON w.item_id = t.item_id
            LEFT JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = TRUE AND d.dress_id IS NULL
            ORDER BY w.item_id
            LIMIT 10
        """)
        
        rows = cur.fetchall()
        if rows:
            print(f"Found {len(rows)} items (showing first 10):")
            for row in rows:
                print(f"  item_id={row[0]}: has_dress={row[1]}, has_top={row[2]}, top_attrs=({row[3]}, {row[4]}, {row[5]})")
            print("\nCONFIRMED: dress data is in top_attributes!")
        else:
            print("No data found in top_attributes")
        
        print("\n" + "="*80 + "\n")
        
finally:
    conn.close()

