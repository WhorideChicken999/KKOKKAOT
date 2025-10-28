-- 사용자 아이템 확인
SELECT 
    user_id,
    COUNT(*) as item_count,
    COUNT(CASE WHEN is_default = TRUE THEN 1 END) as default_items,
    COUNT(CASE WHEN is_default = FALSE OR is_default IS NULL THEN 1 END) as user_items
FROM wardrobe_items 
GROUP BY user_id
ORDER BY user_id;
