// components/figma/homescreen.tsx
import {
  Bell,
  HelpCircle,
  LogOut,
  Menu,
  Settings,
  User as UserIcon,
  X,
  Heart,
} from 'lucide-react-native';
import React, { useState, useEffect, useCallback } from 'react';
import {
  Image,
  ImageBackground,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import BottomNavBar from '../common/BottomNavBar';
import { MainScreen } from '../../App';

const API_BASE_URL = 'https://loyd-extemporaneous-annalise.ngrok-free.dev';

type HomeScreenProps = {
  onNavigate: (step: MainScreen) => void;
  userName?: string;
  onLogout: () => void;
};

export default function HomeScreen({
  onNavigate,
  userName = '주인님',
  onLogout,
}: HomeScreenProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [todayWeather, setTodayWeather] = useState({ 
    temp: '22°C', 
    condition: '맑음', 
    recommendation: '가벼운 레이어드 스타일링 추천' 
  });

  // 날씨 데이터 불러오기
  const fetchWeather = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/weather?city=Seoul`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setTodayWeather({
            temp: `${data.temperature}°C`,
            condition: data.description,
            recommendation: data.style_tip
          });
        }
      }
    } catch (error) {
      console.error('❌ 날씨 데이터 로드 실패:', error);
    }
  }, []);

  useEffect(() => {
    fetchWeather();
  }, [fetchWeather]);
  const recentOutfits = [
    { id: 1, image: 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=500', likes: 24, style: '엘레강스' },
    { id: 2, image: 'https://images.unsplash.com/photo-1552374196-c4e7ffc6e126?w=500', likes: 18, style: '스트릿' },
    { id: 3, image: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=500', likes: 32, style: '미니멀' },
  ];
  const quickActions: Array<{ title: string; subtitle: string; image: string; action: MainScreen }> = [
    { title: '오늘 뭐 입지?', subtitle: '상황별 추천', image: 'https://images.unsplash.com/photo-1608680480325-d3ec3cdf7e60?w=500', action: 'daily-outfit' },
    { title: '내 옷장', subtitle: '관리 & 정리', image: 'https://images.unsplash.com/photo-1603798125914-7b5d27789248?w=500', action: 'wardrobe-management' },
    { title: '스타일 분석', subtitle: '취향 리포트', image: 'https://images.unsplash.com/photo-1651335794520-fb1066d93a84?w=500', action: 'style-analysis' },
    { title: '쇼핑 추천', subtitle: '맞춤 제안', image: 'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=500', action: 'shopping' },
  ];

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.headerWrap}>
        <View style={styles.headerInner}>
          <View>
            <View style={styles.rowStart}>
              <Image 
                source={require('../../assets/home_logo.jpg')}
                style={styles.brandLogo}
                resizeMode="contain"
              />
              <View style={styles.headerDivider} />
              <View style={styles.textContainer}>
                <Text style={styles.hello}>안녕하세요, {userName}님</Text>
                <Text style={styles.subMuted}>{todayWeather.temp} · {todayWeather.condition}</Text>
              </View>
            </View>
          </View>
          <Pressable style={styles.iconBtn} onPress={() => setMenuOpen(true)}>
            <Menu size={20} color="#111" />
          </Pressable>
        </View>
      </View>
      <ScrollView contentContainerStyle={styles.scrollPad}>
        <Pressable style={styles.heroCard} onPress={() => onNavigate('today-curation')}>
          <ImageBackground
            source={{ uri: 'https://images.unsplash.com/photo-1551910021-c59ed3bf006b?w=1200' }}
            style={styles.heroBg}
            imageStyle={styles.heroBgImg}
          >
            <View style={styles.heroOverlay} />
            <View style={styles.heroBody}>
              <Text style={styles.heroTitle}>오늘의 추천</Text>
              <Text style={styles.heroDesc}>{todayWeather.recommendation}</Text>
              <View style={styles.whiteBtn}>
                <Text style={styles.whiteBtnText}>추천 보기</Text>
              </View>
            </View>
          </ImageBackground>
        </Pressable>
        <View style={{ marginTop: 24 }}>
          <Text style={styles.sectionTitle}>빠른 바로가기</Text>
          <View style={styles.grid2}>
            {quickActions.map((qa) => (
              <Pressable key={qa.title} onPress={() => onNavigate(qa.action)} style={styles.card}>
                <ImageBackground source={{ uri: qa.image }} style={styles.cardBg} imageStyle={{borderRadius: 8}}>
                  <View style={styles.cardDim} />
                  <View style={styles.cardBody}>
                    <Text style={styles.cardTitle}>{qa.title}</Text>
                    <Text style={styles.cardSub}>{qa.subtitle}</Text>
                  </View>
                </ImageBackground>
              </Pressable>
            ))}
          </View>
        </View>
        <View style={{ marginTop: 24 }}>
          <View style={styles.rowBetween}>
            <Text style={styles.sectionTitle}>최근 스타일링</Text>
            <Pressable onPress={() => onNavigate('recent-styling')}>
              <Text style={styles.ghostLink}>전체보기</Text>
            </Pressable>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{paddingRight: 16}}>
            <View style={{ flexDirection: 'row', gap: 12, paddingVertical: 12 }}>
              {recentOutfits.map((o) => (
                <Pressable key={o.id} style={styles.thumbCard} onPress={() => {}}>
                  <Image source={{ uri: o.image }} style={styles.thumbImg} />
                  <View style={styles.thumbShade} />
                  <View style={styles.likeBadge}>
                    <Heart size={12} color="#000" fill="#FFF" />
                    <Text style={styles.likeBadgeText}>{o.likes}</Text>
                  </View>
                  <View style={styles.styleBadge}>
                    <Text style={styles.styleBadgeText}>{o.style}</Text>
                  </View>
                </Pressable>
              ))}
            </View>
          </ScrollView>
        </View>
      </ScrollView>
      
      <Modal
        visible={menuOpen}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setMenuOpen(false)}
      >
        <Pressable style={styles.modalBackdrop} onPress={() => setMenuOpen(false)}>
            <Pressable style={styles.menuContainer} onPress={() => {}}>
                <View style={styles.menuHeader}>
                    <Text style={styles.menuUser}>{userName}님</Text>
                    <Pressable style={styles.iconBtn} onPress={() => setMenuOpen(false)}>
                        <X size={20} color="#6B7280" />
                    </Pressable>
                </View>
                <View style={styles.menuBody}>
                  <Pressable
                    style={styles.menuRow}
                    onPress={() => { setMenuOpen(false); onNavigate('myinfo'); }}>
                    <UserIcon size={20} color="#4B5563"/>
                    <Text style={styles.menuText}>내 정보</Text>
                  </Pressable>

                  <Pressable
                    style={styles.menuRow}
                    onPress={() => { setMenuOpen(false); onNavigate('settings'); }}>
                    <Settings size={20} color="#4B5563"/>
                    <Text style={styles.menuText}>설정</Text>
                  </Pressable>

                  <Pressable
                    style={styles.menuRow}
                    onPress={() => { setMenuOpen(false); onNavigate('notifications'); }}>
                    <Bell size={20} color="#4B5563"/>
                    <Text style={styles.menuText}>알림</Text>
                  </Pressable>

                  <Pressable
                    style={styles.menuRow}
                    onPress={() => { setMenuOpen(false); onNavigate('support'); }}>
                    <HelpCircle size={20} color="#4B5563"/>
                    <Text style={styles.menuText}>고객센터</Text>
                  </Pressable>
                </View>
                <Pressable style={styles.logoutBtn} onPress={onLogout}>
                    <LogOut size={20} color="#DC2626"/>
                    <Text style={styles.logoutText}>로그아웃</Text>
                </Pressable>
            </Pressable>
        </Pressable>
      </Modal>

      <BottomNavBar activeScreen="home" onNavigate={onNavigate} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#FFFFFF' },
  headerWrap: { borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#F3F4F6', backgroundColor: '#FFFFFFE6' },
  headerInner: { paddingHorizontal: 16, paddingVertical: 12, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  brand: { fontSize: 22, fontWeight: '600', letterSpacing: 0.5, color: '#000' },
  brandLogo: { 
    width: 100, 
    height: 36, 
  },
  headerDivider: { width: 1, height: 20, backgroundColor: '#D1D5DB', marginLeft: 0, marginRight: 8 },
  hello: { fontSize: 16, color: '#111827' },
  subMuted: { fontSize: 12, color: '#6B7280', marginTop: 2 },
  iconBtn: { padding: 4 },
  rowStart: { flexDirection: 'row', alignItems: 'center' },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  scrollPad: { paddingHorizontal: 16, paddingBottom: 24 },
  heroCard: { borderRadius: 12, overflow: 'hidden', height: 200, shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 10, elevation: 3 },
  heroBg: { flex: 1 },
  heroBgImg: { resizeMode: 'cover' },
  heroOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.4)' },
  heroBody: { flex: 1, padding: 20, justifyContent: 'center' },
  heroTitle: { color: '#fff', fontSize: 24, fontWeight: '600' },
  heroDesc: { color: 'rgba(255,255,255,0.9)', marginTop: 10, fontSize: 14, lineHeight: 20 },
  whiteBtn: { marginTop: 16, backgroundColor: '#fff', paddingVertical: 10, paddingHorizontal: 18, alignSelf: 'flex-start', borderRadius: 8 },
  whiteBtnText: { color: '#111', fontWeight: '600' },
  sectionTitle: { fontSize: 18, fontWeight: '600', color: '#111' },
  grid2: { marginTop: 12, flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: 12 },
  card: { width: '48.5%', height: 140, backgroundColor: '#fff', overflow: 'hidden', borderRadius: 8, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 8, elevation: 2 },
  cardBg: { flex: 1 },
  cardDim: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.4)', borderRadius: 8 },
  cardBody: { padding: 12, justifyContent: 'flex-start' },
  cardTitle: { color: '#fff', fontWeight: '600', fontSize: 16 },
  cardSub: { color: 'rgba(255,255,255,0.85)', marginTop: 4, fontSize: 12 },
  thumbCard: { width: 128, height: 160, borderRadius: 8, overflow: 'hidden', backgroundColor: '#E5E7EB' },
  thumbImg: { width: '100%', height: '100%' },
  thumbShade: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.2)' },
  likeBadge: { position: 'absolute', top: 8, right: 8, backgroundColor: 'rgba(255,255,255,0.9)', borderRadius: 12, paddingHorizontal: 6, height: 20, flexDirection: 'row', alignItems: 'center', gap: 4 },
  likeBadgeText: { fontSize: 11, color: '#000', fontWeight: '600' },
  styleBadge: { position: 'absolute', left: 8, bottom: 8, backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 6, paddingVertical: 2, borderRadius: 10 },
  styleBadgeText: { color: '#fff', fontSize: 11 },
  ghostLink: { color: '#6B7280', fontSize: 14 },
  modalBackdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'center', alignItems: 'center', padding: 20 },
  menuContainer: { width: '100%', backgroundColor: '#FFFFFF', borderRadius: 16, padding: 16, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.25, shadowRadius: 3.84, elevation: 5, gap: 8 },
  menuHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingBottom: 12, borderBottomWidth: 1, borderBottomColor: '#F3F4F6' },
  menuUser: { fontSize: 18, fontWeight: 'bold' },
  menuBody: { gap: 8 },
  menuRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 10, paddingHorizontal: 4 },
  menuText: { fontSize: 16, color: '#374151' },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: '#FEE2E2', paddingVertical: 12, borderRadius: 8, marginTop: 8 },
  logoutText: { color: '#DC2626', fontWeight: '600', fontSize: 16 }
});