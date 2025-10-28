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
        print("Checking remaining 3 dress items")
        print("="*80 + "\n")
        
        cur.execute("""
            SELECT w.item_id, w.has_dress, w.has_top, w.has_bottom, w.has_outer
            FROM wardrobe_items w
            LEFT JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = TRUE AND d.dress_id IS NULL
        """)
        
        remaining = cur.fetchall()
        print(f"Found {len(remaining)} items:")
        
        for item in remaining:
            item_id = item[0]
            print(f"\n  item_id={item_id}:")
            print(f"    has_dress={item[1]}, has_top={item[2]}, has_bottom={item[3]}, has_outer={item[4]}")
            
            # top_attributes 확인
            cur.execute("""
                SELECT category, color, fit
                FROM top_attributes
                WHERE item_id = %s
            """, (item_id,))
            top = cur.fetchone()
            if top:
                print(f"    top_attributes: {top[0]}, {top[1]}, {top[2]}")
            else:
                print(f"    top_attributes: NOT FOUND")
            
            # bottom_attributes 확인
            cur.execute("""
                SELECT category, color, fit
                FROM bottom_attributes
                WHERE item_id = %s
            """, (item_id,))
            bottom = cur.fetchone()
            if bottom:
                print(f"    bottom_attributes: {bottom[0]}, {bottom[1]}, {bottom[2]}")
            else:
                print(f"    bottom_attributes: NOT FOUND")
            
            # outer_attributes 확인
            cur.execute("""
                SELECT category, color, fit
                FROM outer_attributes
                WHERE item_id = %s
            """, (item_id,))
            outer = cur.fetchone()
            if outer:
                print(f"    outer_attributes: {outer[0]}, {outer[1]}, {outer[2]}")
            else:
                print(f"    outer_attributes: NOT FOUND")
        
        print("\n" + "="*80 + "\n")
        
finally:
    conn.close()

