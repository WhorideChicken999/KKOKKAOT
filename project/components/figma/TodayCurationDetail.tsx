// project/components/figma/TodayCurationDetail.tsx
import { Bookmark, Heart, Share2 } from 'lucide-react-native';
import React, { useMemo, useState } from 'react';
import {
    Alert,
    Image,
    Pressable,
    ScrollView,
    StyleSheet,
    Text,
    View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AppHeader from '../common/AppHeader'; 
import BottomNavBar from '../common/BottomNavBar';

type NavigationStep =
  | 'today-curation'
  | 'daily-outfit'
  | 'wardrobe-management'
  | 'style-analysis'
  | 'shopping'
  | 'virtual-fitting';

export type TodayCurationDetailProps = {
  onBack: () => void;
  onNavigate: (step: NavigationStep) => void;
};

type Outfit = {
  id: number;
  image: string;
  title: string;
  description: string;
  temperature: string;
  occasion: string;
  likes: number;
  liked?: boolean;
};

const Badge = ({
  label,
  light,
  outline,
}: {
  label: string;
  light?: boolean;
  outline?: boolean;
}) => (
  <View style={[ styles.badge, light && styles.badgeLight, outline && styles.badgeOutline ]}>
    <Text style={[ styles.badgeText, (light || outline) ? styles.badgeTextDark : styles.badgeTextLight ]}>
      {label}
    </Text>
  </View>
);

export default function TodayCurationDetail({ onBack, onNavigate }: TodayCurationDetailProps) {
  const [bookmarked, setBookmarked] = useState(false);

  const initialOutfits = useMemo<Outfit[]>(() => [
      { id: 1, image: 'https://images.unsplash.com/photo-1544022613-e87ca75a784a?auto=format&fit=crop&w=800&q=80', title: '모던 레이어드 룩', description: '가벼운 니트와 트렌치코트의 완벽한 조합', temperature: '18-22°C', occasion: '데일리/오피스', likes: 127, liked: false },
      { id: 2, image: 'https://images.unsplash.com/photo-1445205170230-053b83016050?auto=format&fit=crop&w=800&q=80', title: '캐주얼 시크', description: '편안하면서도 세련된 일상 스타일링', temperature: '20-25°C', occasion: '캐주얼/데이트', likes: 89, liked: true },
    ], []
  );

  const [outfits, setOutfits] = useState<Outfit[]>(initialOutfits);

  const toggleLike = (id: number) => {
    setOutfits((prev) =>
      prev.map((o) =>
        o.id === id ? { ...o, liked: !o.liked, likes: o.liked ? o.likes - 1 : o.likes + 1, } : o
      )
    );
  };
  
  const handleNavigation = (screen: string) => {
    const map: Record<string, NavigationStep> = {
      Home: 'today-curation',
      Wardrobe: 'wardrobe-management',
      Analysis: 'style-analysis',
      Fitting: 'virtual-fitting',
      Shopping: 'shopping',
    };
    if (map[screen]) {
      onNavigate(map[screen]);
    }
  };

  const HeaderRightAction = (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
      <Pressable onPress={() => setBookmarked((b) => !b)} style={styles.iconBtn}>
        <Bookmark size={20} color={bookmarked ? '#111' : '#6B7280'} fill={bookmarked ? '#111' : 'none'} />
      </Pressable>
      <Pressable onPress={() => Alert.alert('공유', '공유 기능은 이후 연결됩니다.')} style={styles.iconBtn}>
        <Share2 size={20} color="#6B7280" />
      </Pressable>
    </View>
  );

  return (
    <SafeAreaView style={styles.safe}>
      <AppHeader
        title="오늘의 큐레이션"
        subtitle="22°C · 맑음"
        onBack={onBack}
        rightAction={HeaderRightAction}
      />

      <ScrollView contentContainerStyle={styles.screenPad}>
        {/* ... (이하 코드는 변경 없음) ... */}
      </ScrollView>

      <BottomNavBar onNavigate={handleNavigation} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#FFFFFF' },
  iconBtn: { padding: 8, borderRadius: 8 },
  screenPad: { padding: 16, gap: 24, paddingBottom: 32 },
  heroCard: { backgroundColor: '#1F2937', borderRadius: 12, padding: 20 },
  heroTitle: { color: '#FFFFFF', fontSize: 22, fontWeight: '600' },
  heroDesc: { color: 'rgba(255,255,255,0.9)', fontSize: 14, lineHeight: 20, marginTop: 8 },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  badgeLight: { backgroundColor: 'rgba(255,255,255,0.2)', borderWidth: StyleSheet.hairlineWidth, borderColor: 'rgba(255,255,255,0.3)' },
  badgeOutline: { backgroundColor: '#FFFFFF', borderWidth: StyleSheet.hairlineWidth, borderColor: '#E5E7EB' },
  badgeText: { fontSize: 12, fontWeight: '500' },
  badgeTextLight: { color: '#FFFFFF' },
  badgeTextDark: { color: '#111827' },
  sectionTitle: { fontSize: 18, fontWeight: '600', color: '#0B0B0B' },
  countText: { fontSize: 12, color: '#6B7280' },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  cardRow: { backgroundColor: '#FFFFFF', borderRadius: 12, overflow: 'hidden', shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 12, elevation: 2, flexDirection: 'row', borderWidth: 1, borderColor: '#F3F4F6' },
  thumb: { width: 128, height: 'auto', backgroundColor: '#F3F4F6' },
  cardContent: { flex: 1, padding: 16, justifyContent: 'space-between' },
  itemTitle: { fontSize: 16, color: '#111827', fontWeight: '600' },
  itemDesc: { fontSize: 13, color: '#4B5563', marginTop: 4 },
  likesText: { fontSize: 13, color: '#6B7280' },
  ghostBtn: { paddingHorizontal: 12, paddingVertical: 8, backgroundColor: '#FFFFFF', borderWidth: StyleSheet.hairlineWidth, borderColor: '#D1D5DB', borderRadius: 8 },
  ghostText: { color: '#111827', fontSize: 13, fontWeight: '600' },
  cardSoft: { backgroundColor: '#F9FAFB', padding: 16, borderRadius: 12, borderWidth: 1, borderColor: '#F3F4F6' },
  tipRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  bullet: { width: 6, height: 6, borderRadius: 999, backgroundColor: '#111', marginTop: 6 },
  tipText: { color: '#4B5563', fontSize: 13, flex: 1 },
});