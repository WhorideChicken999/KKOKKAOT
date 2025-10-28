// components/figma/WardrobeSetup.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  Pressable,
  ScrollView,
  Alert as RNAlert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { Camera, X } from 'lucide-react-native';

export default function WardrobeSetup({
  onComplete,
  onSkip,
}: {
  onComplete: () => void;
  onSkip: () => void;
}) {
  const [images, setImages] = useState<string[]>([]);

  async function ensureCamera() {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      RNAlert.alert('권한 필요', '카메라 권한을 허용해줘.');
      return false;
    } 
    return true;
  }

  async function ensureLibrary() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      RNAlert.alert('권한 필요', '사진 라이브러리 권한을 허용해줘.');
      return false;
    }
    return true;
  }

  async function pickFromCamera() {
    if (!(await ensureCamera())) return;
    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.9,
    });
    if (!res.canceled) {
      const uris = res.assets.map((a) => a.uri);
      setImages((prev) => [...prev, ...uris]);
    }
  }

  async function pickFromLibrary() {
    if (!(await ensureLibrary())) return;
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.9,
      // iOS/Web에서만 다중선택 지원. Android는 한 장만 반환될 수 있음.
      allowsMultipleSelection: true,
      selectionLimit: 10,
    });
    if (!res.canceled) {
      const uris = res.assets.map((a) => a.uri);
      setImages((prev) => [...prev, ...uris]);
    }
  }

  function removeAt(index: number) {
    setImages((prev) => prev.filter((_, i) => i !== index));
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        {/* 헤더 */}
        <View style={styles.header}>
          <Text style={styles.title}>가상 옷장</Text>
          <View style={styles.line} />
          <Text style={styles.sub}>보유하신 옷들을 등록하여 맞춤 추천을 받으세요</Text>
        </View>

        {/* 카드 */}
        <View style={styles.card}>
          {/* 상단 타이틀/개수 */}
          <View style={styles.rowBetween}>
            <Text style={styles.sectionTitle}>옷 등록하기</Text>
            <Text style={styles.countText}>{images.length}개</Text>
          </View>

          {/* 썸네일 그리드 */}
          {images.length > 0 && (
            <View style={styles.grid}>
              {images.map((uri, idx) => (
                <View key={uri + idx} style={styles.gridItem}>
                  <Image source={{ uri }} style={styles.gridImg} />
                  <Pressable onPress={() => removeAt(idx)} style={styles.removeBtn}>
                    <X size={12} color="#FFF" />
                  </Pressable>
                </View>
              ))}
            </View>
          )}

          {/* 플레이스홀더 */}
          <View style={styles.placeholder}>
            <View style={styles.placeholderIcon}>
              <Camera size={32} color="#9CA3AF" />
            </View>
            <View style={{ alignItems: 'center', gap: 4 }}>
              <Text style={styles.phMain}>옷을 촬영하여 AI가 자동으로 분류합니다</Text>
              <Text style={styles.phSub}>상의, 하의, 아우터를 자동 인식</Text>
            </View>
          </View>

          {/* 액션 버튼 */}
          <View style={{ gap: 10 }}>
            <Pressable onPress={pickFromCamera} style={[styles.btn, styles.btnPrimary]}>
              <Camera size={18} color="#FFF" />
              <Text style={styles.btnPrimaryText}>옷 촬영하기</Text>
            </Pressable>
            <Pressable onPress={pickFromLibrary} style={[styles.btn, styles.btnOutline]}>
              <Text style={styles.btnOutlineText}>앨범에서 선택</Text>
            </Pressable>
          </View>

          {/* 가이드 박스 */}
          <View style={styles.guideBox}>
            <Text style={styles.guideTitle}>촬영 가이드</Text>
            {['옷을 평평하게 펼쳐서 촬영하세요', '밝은 배경에서 촬영하면 더 정확해요', 'AI가 자동으로 상의/하의/아우터를 분류합니다', '나중에 언제든 추가할 수 있습니다'].map(
              (t) => (
                <Text key={t} style={styles.guideItem}>
                  • {t}
                </Text>
              )
            )}
          </View>

          {/* 완료/건너뛰기 */}
          <View style={{ gap: 10 }}>
            {images.length > 0 && (
              <Pressable onPress={onComplete} style={[styles.btn, styles.btnPrimary]}>
                <Text style={styles.btnPrimaryText}>옷장 등록 완료 ({images.length}개)</Text>
              </Pressable>
            )}
            <Pressable onPress={onSkip} style={[styles.btn, styles.btnOutline]}>
              <Text style={styles.btnOutlineText}>나중에 등록하기</Text>
            </Pressable>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F9FAFB' },
  container: { padding: 16, paddingTop: 24 },
  header: { alignItems: 'center', gap: 6, marginBottom: 16 },
  title: { fontSize: 20, color: '#111827', letterSpacing: 1, fontFamily: 'PlayfairDisplay-SemiBold' },
  line: { width: 32, height: 1, backgroundColor: '#D1D5DB' },
  sub: { fontSize: 12, color: '#6B7280', fontFamily: 'Inter' },

  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 3,
    padding: 16,
    gap: 16,
  },

  rowBetween: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  sectionTitle: { fontSize: 16, color: '#111827', fontFamily: 'Inter', fontWeight: '300' },
  countText: { fontSize: 13, color: '#6B7280', fontFamily: 'Inter' },

  grid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', gap: 10 },
  gridItem: { width: '32%', aspectRatio: 1, position: 'relative', backgroundColor: '#F3F4F6', overflow: 'hidden' },
  gridImg: { width: '100%', height: '100%', resizeMode: 'cover' },
  removeBtn: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#111',
    alignItems: 'center',
    justifyContent: 'center',
  },

  placeholder: {
    borderWidth: 1,
    borderColor: '#E5E7EB',
    backgroundColor: '#F3F4F6',
    aspectRatio: 4 / 3,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  placeholderIcon: { width: 64, height: 64, borderRadius: 32, backgroundColor: '#E5E7EB', alignItems: 'center', justifyContent: 'center' },
  phMain: { fontSize: 13, color: '#4B5563', fontFamily: 'Inter' },
  phSub: { fontSize: 11, color: '#6B7280', fontFamily: 'Inter' },

  btn: { paddingVertical: 14, alignItems: 'center', justifyContent: 'center', borderRadius: 8, flexDirection: 'row', gap: 8 },
  btnPrimary: { backgroundColor: '#111111' },
  btnPrimaryText: { color: '#FFFFFF', fontSize: 14, fontFamily: 'Inter' },
  btnOutline: { borderWidth: 1, borderColor: '#E5E7EB', backgroundColor: '#FFFFFF' },
  btnOutlineText: { color: '#111111', fontSize: 14, fontFamily: 'Inter' },

  guideBox: { backgroundColor: '#F9FAFB', borderLeftWidth: 2, borderLeftColor: '#9CA3AF', padding: 12, gap: 4 },
  guideTitle: { fontSize: 13, color: '#111827', fontFamily: 'Inter' },
  guideItem: { fontSize: 12, color: '#6B7280', fontFamily: 'Inter' },
});
