import psycopg2
from psycopg2.extras import Json

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
        print("Fixing remaining 3 dress items (442, 455, 457)")
        print("="*80 + "\n")
        
        # 이 3개는 has_outer=True이고 has_dress=True
        # outer_attributes에 데이터가 있으니 dress_attributes로 복사
        remaining_ids = [442, 455, 457]
        
        for item_id in remaining_ids:
            # outer_attributes에서 데이터 가져오기
            cur.execute("""
                SELECT category, color, fit, materials,
                       category_confidence, color_confidence, fit_confidence
                FROM outer_attributes
                WHERE item_id = %s
            """, (item_id,))
            
            outer_data = cur.fetchone()
            if outer_data:
                print(f"item_id={item_id}:")
                print(f"  Copying from outer_attributes: {outer_data[0]}, {outer_data[1]}, {outer_data[2]}")
                
                # dress_attributes로 복사
                cur.execute("""
                    INSERT INTO dress_attributes (
                        item_id, category, color, fit, materials,
                        category_confidence, color_confidence, fit_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    outer_data[0],  # category
                    outer_data[1],  # color
                    outer_data[2],  # fit
                    Json(outer_data[3]) if outer_data[3] else Json([]),  # materials
                    outer_data[4],  # category_confidence
                    outer_data[5],  # color_confidence
                    outer_data[6]   # fit_confidence
                ))
                
                print(f"  => Copied to dress_attributes")
            else:
                print(f"item_id={item_id}: No data in outer_attributes!")
        
        # COMMIT
        print("\n" + "="*80)
        print("Committing...")
        conn.commit()
        print("=> SUCCESS!")
        
        # VERIFICATION
        print("\n" + "="*80)
        print("FINAL VERIFICATION:")
        print("="*80)
        
        cur.execute("""
            SELECT COUNT(*)
            FROM wardrobe_items w
            LEFT JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = TRUE AND d.dress_id IS NULL
        """)
        still_missing = cur.fetchone()[0]
        
        print(f"Missing dress_attributes: {still_missing}")
        
        if still_missing == 0:
            print("\nPERFECT! All dress items fixed!")
        
        print("="*80 + "\n")
        
finally:
    conn.close()

