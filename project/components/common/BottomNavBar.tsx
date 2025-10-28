// components/common/BottomNavBar.tsx
import {
  BarChart3,
  Home,
  ShoppingBag,
  Shirt as ShirtIcon,
  User as UserIcon,
  MessageCircle,
  Scan,
} from 'lucide-react-native';
import React from 'react';
import { View, Text, StyleSheet, Pressable, Alert as RNAlert } from 'react-native';
import { MainScreen } from '../../App';

type BottomNavBarProps = {
  activeScreen?: MainScreen;
  onNavigate?: (screen: MainScreen) => void;
  disabled?: boolean;
};

const navItems: { icon: any, title: string, screen: MainScreen }[] = [
  { icon: Home, title: '홈', screen: 'home' },
  { icon: ShirtIcon, title: '옷장', screen: 'wardrobe-management' },
  { icon: BarChart3, title: '분석', screen: 'style-analysis' },
  { icon: MessageCircle, title: 'AI', screen: 'llm-chat' },
  { icon: Scan, title: '피팅', screen: 'virtual-fitting' },
  { icon: ShoppingBag, title: '쇼핑', screen: 'shopping' },
];

export default function BottomNavBar({ activeScreen, onNavigate, disabled = false }: BottomNavBarProps) {
  const handlePress = (screen: MainScreen) => {
    if (disabled) {
      RNAlert.alert('안내', '프로필 설정을 먼저 완료해주세요.');
      return;
    }
    if (onNavigate) {
      onNavigate(screen);
    }
  };

  return (
    <View style={styles.bottomBar}>
      <View style={styles.bottomInner}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = !disabled && activeScreen === item.screen;
          const color = isActive ? '#111' : '#9CA3AF';

          return (
            <Pressable key={item.title} style={styles.bottomItem} onPress={() => handlePress(item.screen)}>
              <Icon size={20} color={color} />
              <Text style={[styles.bottomLabel, { color }]}>{item.title}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  bottomBar: {
    // ✅ 네비게이션 바를 화면 하단에 고정
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    // ✅ zIndex를 높게 설정하여 다른 컴포넌트에 가려지지 않게 함
    zIndex: 100,
    
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#E5E7EB',
  },
  bottomInner: {
    paddingHorizontal: 16,
    paddingTop: 8,
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingBottom: 24,
  },
  bottomItem: { alignItems: 'center', justifyContent: 'center', paddingVertical: 6, gap: 2, flex: 1 },
  bottomLabel: { fontSize: 11 },
});