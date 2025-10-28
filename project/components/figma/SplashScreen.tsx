// components/figma/SplashScreen.tsx
import React from 'react';
import {
  Dimensions,
  ImageBackground,
  Pressable,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

// **참고:** '시작화면.png' 파일을 프로젝트 루트의 'assets' 폴더 안에 넣고 'splash.png'로 이름을 변경해주세요.
const splashImage = require('../../assets/images/splash-icon.png'); 

const { height: H } = Dimensions.get('window');
const SHIFT_Y = Math.round(H * 0.02);
const SCALE   = 1.12;

export type SplashScreenProps = { onGetStarted: () => void };

export default function SplashScreen({ onGetStarted }: SplashScreenProps) {
  return (
    <View style={styles.root}>
      <StatusBar translucent backgroundColor="transparent" barStyle="dark-content" />
      <ImageBackground
        source={splashImage}
        resizeMode="cover"
        style={styles.bgAbs}
        imageStyle={{ transform: [{ translateY: SHIFT_Y }, { scale: SCALE }] }}
      />
      <SafeAreaView edges={['bottom']} style={styles.overlay}>
        <View style={styles.bottomWrap}>
          <View style={styles.card}>
            <Text style={styles.title}>꼬까옷에 오신 것을{'\n'}환영합니다</Text>
            <Text style={styles.sub}>AI가 당신의 체형과 취향을 분석하여{'\n'}완벽한 스타일링을 제안해드립니다</Text>
            <Pressable onPress={onGetStarted} style={styles.cta}>
              <Text style={styles.ctaText}>시작하기</Text>
            </Pressable>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: 'transparent' },
  bgAbs: { ...StyleSheet.absoluteFillObject },
  overlay: { flex: 1 },
  bottomWrap: { padding: 16, flex: 1, justifyContent: 'flex-end' },
  card: {
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderRadius: 20,
    padding: 24,
    shadowColor: '#000',
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 8,
  },
  title: { textAlign: 'center', fontSize: 22, fontWeight: 'bold', color: '#111827', marginBottom: 12 },
  sub: { textAlign: 'center', fontSize: 14, lineHeight: 22, color: '#374151', marginBottom: 24 },
  cta: { backgroundColor: '#111111', borderRadius: 14, height: 52, alignItems: 'center', justifyContent: 'center' },
  ctaText: { color: '#FFFFFF', fontSize: 16, fontWeight: '600' },
});