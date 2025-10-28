// project/components/figma/ShoppingRecommendations.tsx
import { ExternalLink, Filter, Heart, Search, ShoppingBag, Star } from 'lucide-react-native';
import React, { useMemo, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AppHeader from '../common/AppHeader';
import BottomNavBar from '../common/BottomNavBar';
import { MainScreen } from '../../App';

export type ShoppingRecommendationsProps = {
  onBack: () => void;
  onNavigate: (step: MainScreen) => void;
};

export default function ShoppingRecommendations({ onBack, onNavigate }: ShoppingRecommendationsProps) {
  const [selectedCategory, setSelectedCategory] = useState<'recommended' | 'trending' | 'missing' | 'seasonal'>('recommended');
  const [priceRange, setPriceRange] = useState<'all' | 'under50' | '50to100' | '100to200' | 'over200'>('all');

  const categories = useMemo(
    () => [
      { id: 'recommended', name: 'AI 추천', icon: '🎯' },
      { id: 'trending', name: '트렌드', icon: '🔥' },
      { id: 'missing', name: '부족한 아이템', icon: '📝' },
      { id: 'seasonal', name: '시즌 필수', icon: '🌸' },
    ],
    []
  );

  const priceRanges = useMemo(
    () => [
      { id: 'all', name: '전체' },
      { id: 'under50', name: '5만원 이하' },
      { id: '50to100', name: '5-10만원' },
      { id: '100to200', name: '10-20만원' },
      { id: 'over200', name: '20만원 이상' },
    ],
    []
  );

  const recommendedItems = useMemo(
    () => [
      {
        id: 1,
        image:
          'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=800&q=80',
        name: '클래식 트렌치 코트',
        brand: 'Burberry',
        price: '2,890,000원',
        originalPrice: '3,200,000원',
        discount: 10,
        rating: 4.8,
        reviews: 127,
        matchScore: 95,
        reason: '주인님의 미니멀 스타일과 완벽 매치',
        tags: ['베스트셀러', '리뷰 좋음'],
        inStock: true,
        fastShipping: true,
      },
      {
        id: 2,
        image:
          'https://images.unsplash.com/photo-1544441893-675973e31985?auto=format&fit=crop&w=800&q=80',
        name: '코튼 화이트 셔츠',
        brand: 'COS',
        price: '129,000원',
        originalPrice: null,
        discount: 0,
        rating: 4.6,
        reviews: 89,
        matchScore: 92,
        reason: '옷장의 블랙 팬츠와 완벽한 조합',
        tags: ['신상품'],
        inStock: true,
        fastShipping: false,
      },
      {
        id: 3,
        image:
          'https://images.unsplash.com/photo-1549062572-544a64fb0c56?auto=format&fit=crop&w=800&q=80',
        name: '미니멀 레더 백',
        brand: 'Mansur Gavriel',
        price: '450,000원',
        originalPrice: '520,000원',
        discount: 15,
        rating: 4.9,
        reviews: 203,
        matchScore: 88,
        reason: '심플한 디자인으로 데일리 매치 완벽',
        tags: ['한정 세일', '리뷰 좋음'],
        inStock: false,
        fastShipping: false,
      },
    ],
    []
  );

  const filteredItems = recommendedItems;

  const HeaderRightAction = (
    <Pressable style={[ styles.primaryBtn, { paddingHorizontal: 12, paddingVertical: 8, flexDirection: 'row', alignItems: 'center', gap: 6 }]}>
      <ShoppingBag size={16} color="#FFF" />
      <Text style={styles.primaryBtnText}>장바구니</Text>
    </Pressable>
  );

  return (
    <SafeAreaView style={styles.safe}>
      <AppHeader
        title="쇼핑 추천"
        subtitle="AI 맞춤 상품 추천"
        onBack={onBack}
        rightAction={HeaderRightAction}
      />

      <ScrollView contentContainerStyle={styles.screenPad}>
        <View style={{ position: 'relative' }}>
          <Search size={16} color="#9CA3AF" style={styles.searchIcon} />
          <TextInput
            placeholder="원하는 아이템을 검색해보세요..."
            placeholderTextColor="#9CA3AF"
            style={styles.input}
          />
          <Pressable style={styles.filterBtn}>
            <Filter size={16} color="#4B5563" />
          </Pressable>
        </View>

        <View style={{ gap: 8 }}>
          <Text style={styles.sectionLabel}>카테고리</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View style={{ flexDirection: 'row', gap: 8, paddingBottom: 4 }}>
              {categories.map((c) => (
                <Pressable
                  key={c.id}
                  onPress={() => setSelectedCategory(c.id as any)}
                  style={[
                    styles.chip,
                    selectedCategory === c.id ? styles.chipActive : styles.chipIdle,
                  ]}
                >
                  <Text style={[styles.chipText, selectedCategory === c.id && styles.chipTextActive]}>
                    {c.icon} {c.name}
                  </Text>
                </Pressable>
              ))}
            </View>
          </ScrollView>
        </View>

        <View style={{ gap: 12 }}>
          <View style={styles.rowBetween}>
            <Text style={styles.sectionTitle}>
              {categories.find(c => c.id === selectedCategory)?.name || 'AI 맞춤 추천'}
            </Text>
            <Text style={styles.countText}>{filteredItems.length}개 상품</Text>
          </View>

          <View style={{ gap: 12 }}>
            {filteredItems.map((item, index) => (
              <View key={`${item.id}-${index}`} style={styles.cardRow}>
                <View style={styles.thumbWrap}>
                  <Image source={{ uri: item.image }} style={styles.thumb} />
                  {item.discount > 0 && (
                    <View style={styles.discountBadge}>
                      <Text style={styles.badgeTextWhite}>-{item.discount}%</Text>
                    </View>
                  )}
                  {!item.inStock && (
                    <View style={styles.soldoutOverlay}>
                      <Text style={styles.badgeTextWhite}>품절</Text>
                    </View>
                  )}
                </View>

                <View style={styles.cardContent}>
                  <View style={styles.rowBetween}>
                    <View style={{ flex: 1, gap: 2 }}>
                      <Text style={styles.itemName}>{item.name}</Text>
                      <Text style={styles.itemBrand}>{item.brand}</Text>
                    </View>
                    <Pressable hitSlop={8}>
                      <Heart size={16} color="#9CA3AF" />
                    </Pressable>
                  </View>

                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
                    <View style={styles.metaRow}>
                      <Star size={14} color="#F59E0B" fill="#F59E0B" />
                      <Text style={styles.metaText}>{item.rating}</Text>
                      <Text style={styles.metaSub}>({item.reviews})</Text>
                    </View>
                    <View style={styles.metaRow}>
                      <View
                        style={[
                          styles.dot,
                          { backgroundColor: item.matchScore >= 90 ? '#22C55E' : '#EAB308' },
                        ]}
                      />
                      <Text style={styles.metaText}>매치 {item.matchScore}%</Text>
                    </View>
                  </View>

                  <Text style={styles.reasonText}>{item.reason}</Text>

                  <View style={styles.rowBetween}>
                    <View style={{ gap: 4 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                        <Text style={styles.price}>{item.price}</Text>
                        {item.originalPrice ? (
                          <Text style={styles.priceStrike}>{item.originalPrice}</Text>
                        ) : null}
                      </View>
                    </View>

                    <Pressable
                      disabled={!item.inStock}
                      style={[
                        styles.primaryBtn,
                        { paddingHorizontal: 12, paddingVertical: 8, opacity: item.inStock ? 1 : 0.5 },
                      ]}
                    >
                      <Text style={styles.primaryBtnText}>{item.inStock ? '담기' : '품절'}</Text>
                    </Pressable>
                  </View>
                </View>
              </View>
            ))}
          </View>
        </View>

      </ScrollView>

      <BottomNavBar activeScreen="shopping" onNavigate={onNavigate} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#FFFFFF' },
  screenPad: { paddingHorizontal: 16, paddingVertical: 24, gap: 24, paddingBottom: 32 },
  searchIcon: { position: 'absolute', left: 12, top: 14, zIndex: 1 },
  input: {
    backgroundColor: '#F9FAFB',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 8,
    paddingVertical: 10,
    paddingLeft: 40,
    paddingRight: 50,
    fontSize: 14,
    color: '#111827',
  },
  filterBtn: {
    position: 'absolute',
    right: 6,
    top: 6,
    padding: 8,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  sectionLabel: { fontWeight: '600', color: '#111827' },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 99,
    borderWidth: 1,
  },
  chipIdle: { backgroundColor: '#FFFFFF', borderColor: '#E5E7EB' },
  chipActive: { backgroundColor: '#111111', borderColor: '#111111' },
  chipText: { color: '#111827', fontSize: 13 },
  chipTextActive: { color: '#FFFFFF', fontWeight: '600' },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  sectionTitle: { fontSize: 18, fontWeight: '600' },
  countText: { fontSize: 12, color: '#6B7280' },
  cardRow: {
    flexDirection: 'row',
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 2,
    borderWidth: 1,
    borderColor: '#F3F4F6',
  },
  thumbWrap: { width: 96, height: 128, backgroundColor: '#F3F4F6', borderTopLeftRadius: 8, borderBottomLeftRadius: 8, overflow: 'hidden' },
  thumb: { width: '100%', height: '100%' },
  discountBadge: {
    position: 'absolute',
    top: 8,
    left: 8,
    backgroundColor: '#EF4444',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  badgeTextWhite: {
    color: '#FFF',
    fontSize: 11,
    fontWeight: '600',
  },
  soldoutOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardContent: {
    flex: 1,
    padding: 12,
    gap: 8,
  },
  itemName: { fontSize: 14, fontWeight: '600' },
  itemBrand: { fontSize: 12, color: '#6B7280' },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4
  },
  metaText: { fontSize: 12, fontWeight: '600', color: '#111827' },
  metaSub: { fontSize: 12, color: '#9CA3AF' },
  dot: { width: 8, height: 8, borderRadius: 4 },
  reasonText: { fontSize: 12, color: '#4B5563' },
  price: { fontSize: 14, fontWeight: '600' },
  priceStrike: { fontSize: 12, color: '#9CA3AF', textDecorationLine: 'line-through' },
  primaryBtn: { backgroundColor: '#111111', borderRadius: 8 },
  primaryBtnText: { color: '#FFFFFF', fontSize: 13, fontWeight: '600' },
});