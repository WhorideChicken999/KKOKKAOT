// components/figma/UserProfileSetup.tsx
import React, { useMemo, useState } from 'react';
import {
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export type UserProfileSetupProps = {
  // 완료 시 이름 등을 넘겨 App에서 인사말에 사용
  onComplete: (data: {
    name: string;
    gender?: string;
    ageGroup?: string;
    skinTone?: string;
    stylePreferences?: string[];
    measurements?: { height?: string; weight?: string; chest?: string; waist?: string; hip?: string };
  }) => void;
};

const SKIN_TONES = [
  { id: 'cool-fair', name: '쿨 페어', color: '#F7E7CE' },
  { id: 'warm-fair', name: '웜 페어', color: '#F2D7A7' },
  { id: 'cool-medium', name: '쿨 미디움', color: '#E8B887' },
  { id: 'warm-medium', name: '웜 미디움', color: '#D4A574' },
  { id: 'cool-tan', name: '쿨 탄', color: '#C08B5C' },
  { id: 'warm-tan', name: '웜 탄', color: '#A67449' },
];

const STYLE_PREFERENCES = [
  {
    name: '기타',
    description: '다른 카테고리에 속하지 않는 독특한 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/etc.jpg',
  },
  {
    name: '레트로',
    description: '과거 시대의 감성과 스타일을 재해석한 룩',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/retro.jpg',
  },
  {
    name: '로맨틱',
    description: '사랑스럽고 여성스러운 로맨틱 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/romantic.jpg',
  },
  {
    name: '리조트',
    description: '휴양지에서 즐기는 여유롭고 세련된 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/resort.jpg',
  },
  {
    name: '매니시',
    description: '남성적인 요소를 여성스럽게 소화한 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/manish.jpg',
  },
  {
    name: '모던',
    description: '현대적이고 세련된 미니멀 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/modern.jpg',
  },
  {
    name: '밀리터리',
    description: '군복에서 영감을 받은 강인하고 실용적인 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/military.jpg',
  },
  {
    name: '섹시',
    description: '매력적이고 관능적인 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/sexy.jpg',
  },
  {
    name: '소피스트케이티드',
    description: '세련되고 지적인 고급스러운 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/sophisticated.jpg',
  },
  {
    name: '스트리트',
    description: '개성있고 트렌디한 도시 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/street.jpg',
  },
  {
    name: '스포티',
    description: '활동적이고 편안한 스포츠 룩',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/sporty.jpg',
  },
  {
    name: '아방가르드',
    description: '실험적이고 혁신적인 예술적 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/avangard.jpg',
  },
  {
    name: '오리엔탈',
    description: '동양의 전통과 현대가 만나는 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/oriental.jpg',
  },
  {
    name: '웨스턴',
    description: '미국 서부의 카우보이 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/western.jpg',
  },
  {
    name: '젠더리스',
    description: '성별을 초월한 중성적이고 자유로운 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/genderless.jpg',
  },
  {
    name: '컨트리',
    description: '시골의 자연스럽고 편안한 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/country.jpg',
  },
  {
    name: '클래식',
    description: '우아하고 고급스러운 정통 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/classic.jpg',
  },
  {
    name: '키치',
    description: '유머러스하고 장식적인 팝 아트 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/kitsch.jpg',
  },
  {
    name: '톰보이',
    description: '남성적인 요소를 자연스럽게 소화한 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/tomboy.jpg',
  },
  {
    name: '펑크',
    description: '반항적이고 독립적인 록 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/funk.jpg',
  },
  {
    name: '페미닌',
    description: '여성스럽고 우아한 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/feminin.jpg',
  },
  {
    name: '프레피',
    description: '단정하고 깔끔한 아이비리그 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/preppy.jpg',
  },
  {
    name: '히피',
    description: '자유롭고 예술적인 보헤미안 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/hippy.jpg',
  },
  {
    name: '힙합',
    description: '힙합 문화에서 영감을 받은 스트리트 스타일',
    image: 'https://loyd-extemporaneous-annalise.ngrok-free.dev/api/represent-images/hiphop.jpg',
  },
];

export default function UserProfileSetup({ onComplete }: UserProfileSetupProps) {
  const [step, setStep] = useState(1);
  const totalSteps = 5;

  const [form, setForm] = useState({
    name: '',
    gender: '',
    ageGroup: '',
    skinTone: '',
    stylePreferences: [] as string[],
    measurements: { height: '', weight: '', chest: '', waist: '', hip: '' },
  });

  const progress = useMemo(() => (step / totalSteps) * 100, [step]);

  const next = () => {
    if (step < totalSteps) {
      setStep((s) => s + 1);
      return;
    }
    // ✅ 완료: 이름 포함 payload 전달
    onComplete({
      name: form.name.trim(),
      gender: form.gender,
      ageGroup: form.ageGroup,
      skinTone: form.skinTone,
      stylePreferences: form.stylePreferences,
      measurements: form.measurements,
    });
  };
  const back = () => step > 1 && setStep((s) => s - 1);

  const toggleStyle = (name: string) => {
    setForm((prev) => {
      const has = prev.stylePreferences.includes(name);
      return {
        ...prev,
        stylePreferences: has
          ? prev.stylePreferences.filter((v) => v !== name)
          : [...prev.stylePreferences, name],
      };
    });
  };

  const canProceed = useMemo(() => {
    if (step === 1) return form.name.trim().length > 0;
    if (step === 2) return !!form.gender;
    if (step === 3) return !!form.ageGroup;
    if (step === 4) return !!form.skinTone;
    if (step === 5) return form.stylePreferences.length > 0;
    return true;
  }, [step, form]);

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          {/* 헤더 + 프로그레스 */}
          <View style={{ marginBottom: 16 }}>
            <View style={{ alignItems: 'center' }}>
              <Text style={styles.title}>프로필 설정</Text>
              <View style={styles.line} />
            </View>

            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: `${progress}%` }]} />
            </View>
            <Text style={styles.progressText}>
              {step} / {totalSteps}
            </Text>
          </View>

          {/* 카드 */}
          <View style={styles.card}>
            <View style={{ padding: 20 }}>
              {/* STEP 1: 이름 */}
              {step === 1 && (
                <View style={{ gap: 12 }}>
                  <View style={{ alignItems: 'center' }}>
                    <Text style={styles.sectionTitle}>사용하실 이름을 알려주세요</Text>
                    <Text style={styles.sectionSub}>개인화된 서비스를 위해 필요합니다</Text>
                  </View>

                  <TextInput
                    placeholder="닉네임을 입력하세요"
                    value={form.name}
                    onChangeText={(t) => setForm((p) => ({ ...p, name: t }))}
                    placeholderTextColor="#9CA3AF"
                    style={[styles.input, { textAlign: 'center' }]}
                  />
                </View>
              )}

              {/* STEP 2: 성별 */}
              {step === 2 && (
                <View style={{ gap: 12 }}>
                  <View style={{ alignItems: 'center' }}>
                    <Text style={styles.sectionTitle}>성별을 선택해주세요</Text>
                    <Text style={styles.sectionSub}>맞춤 추천을 위해 필요합니다</Text>
                  </View>

                  {[
                    { id: 'female', label: '여성' },
                    { id: 'male', label: '남성' },
                    { id: 'other', label: '기타' },
                  ].map((g) => (
                    <Pressable
                      key={g.id}
                      onPress={() => setForm((p) => ({ ...p, gender: g.id }))}
                      style={[
                        styles.rowBetween,
                        styles.selectRow,
                        form.gender === g.id ? styles.selectRowActive : null,
                      ]}
                    >
                      <Text style={styles.selectLabel}>{g.label}</Text>
                      <View
                        style={[
                          styles.circle,
                          form.gender === g.id ? styles.circleOn : styles.circleOff,
                        ]}
                      />
                    </Pressable>
                  ))}
                </View>
              )}

              {/* STEP 3: 나이대 */}
              {step === 3 && (
                <View style={{ gap: 12 }}>
                  <View style={{ alignItems: 'center' }}>
                    <Text style={styles.sectionTitle}>나이대를 선택해주세요</Text>
                    <Text style={styles.sectionSub}>연령에 맞는 스타일을 제안해드립니다</Text>
                  </View>

                  {['10s', '20s', '30s', '40s', '50s'].map((age) => (
                    <Pressable
                      key={age}
                      onPress={() => setForm((p) => ({ ...p, ageGroup: age }))}
                      style={[
                        styles.rowBetween,
                        styles.selectRow,
                        form.ageGroup === age ? styles.selectRowActive : null,
                      ]}
                    >
                      <Text style={styles.selectLabel}>
                        {age === '10s'
                          ? '10대'
                          : age === '20s'
                          ? '20대'
                          : age === '30s'
                          ? '30대'
                          : age === '40s'
                          ? '40대'
                          : '50대 이상'}
                      </Text>
                      <View
                        style={[
                          styles.circle,
                          form.ageGroup === age ? styles.circleOn : styles.circleOff,
                        ]}
                      />
                    </Pressable>
                  ))}
                </View>
              )}

              {/* STEP 4: 피부톤 */}
              {step === 4 && (
                <View style={{ gap: 12 }}>
                  <View style={{ alignItems: 'center' }}>
                    <Text style={styles.sectionTitle}>본인의 정확한 피부톤을 선택해주세요</Text>
                    <Text style={styles.sectionSub}>정확한 분석을 위해 필요합니다</Text>
                  </View>

                  <View style={styles.grid2}>
                    {SKIN_TONES.map((tone) => (
                      <Pressable
                        key={tone.id}
                        onPress={() => setForm((p) => ({ ...p, skinTone: tone.id }))}
                        style={[
                          styles.skinItem,
                          form.skinTone === tone.id ? styles.skinItemActive : null,
                        ]}
                      >
                        <View style={styles.skinRow}>
                          <View style={[styles.skinDot, { backgroundColor: tone.color }]} />
                          <Text style={styles.skinName}>{tone.name}</Text>
                        </View>
                      </Pressable>
                    ))}
                  </View>
                </View>
              )}

              {/* STEP 5: 스타일 + 신체정보 */}
              {step === 5 && (
                <View style={{ gap: 16 }}>
                  <View style={{ alignItems: 'center' }}>
                    <Text style={styles.sectionTitle}>선호하는 스타일을 선택해주세요</Text>
                    <Text style={styles.sectionSub}>
                      여러 개 선택 가능합니다 ({form.stylePreferences.length}개 선택됨)
                    </Text>
                  </View>

                  <View style={styles.grid2}>
                    {STYLE_PREFERENCES.map((s) => (
                      <Pressable
                        key={s.name}
                        onPress={() => toggleStyle(s.name)}
                        style={[
                          styles.styleCard,
                          form.stylePreferences.includes(s.name) && styles.styleCardActive,
                        ]}
                      >
                        <Image
                          source={{ uri: s.image }}
                          style={styles.styleThumb}
                          resizeMode="cover"
                        />
                        <View style={styles.styleOverlay} />
                        <Text style={styles.styleName}>{s.name}</Text>
                      </Pressable>
                    ))}
                  </View>

                  <View style={styles.tipBox}>
                    <Text style={styles.tipTitle}>신체 정보 (선택사항)</Text>
                    <Text style={styles.tipSub}>작성하시면 더 정확한 분석이 가능합니다</Text>

                    <View style={styles.grid2}>
                      <TextInput
                        placeholder="키 (cm)"
                        keyboardType="numeric"
                        value={form.measurements.height}
                        onChangeText={(t) =>
                          setForm((p) => ({ ...p, measurements: { ...p.measurements, height: t } }))
                        }
                        placeholderTextColor="#9CA3AF"
                        style={styles.input}
                      />
                      <TextInput
                        placeholder="몸무게 (kg)"
                        keyboardType="numeric"
                        value={form.measurements.weight}
                        onChangeText={(t) =>
                          setForm((p) => ({ ...p, measurements: { ...p.measurements, weight: t } }))
                        }
                        placeholderTextColor="#9CA3AF"
                        style={styles.input}
                      />
                      <TextInput
                        placeholder="가슴둘레 (cm)"
                        keyboardType="numeric"
                        value={form.measurements.chest}
                        onChangeText={(t) =>
                          setForm((p) => ({ ...p, measurements: { ...p.measurements, chest: t } }))
                        }
                        placeholderTextColor="#9CA3AF"
                        style={styles.input}
                      />
                      <TextInput
                        placeholder="허리둘레 (cm)"
                        keyboardType="numeric"
                        value={form.measurements.waist}
                        onChangeText={(t) =>
                          setForm((p) => ({ ...p, measurements: { ...p.measurements, waist: t } }))
                        }
                        placeholderTextColor="#9CA3AF"
                        style={styles.input}
                      />
                    </View>
                    <TextInput
                      placeholder="엉덩이둘레 (cm)"
                      keyboardType="numeric"
                      value={form.measurements.hip}
                      onChangeText={(t) =>
                        setForm((p) => ({ ...p, measurements: { ...p.measurements, hip: t } }))
                      }
                      placeholderTextColor="#9CA3AF"
                      style={styles.input}
                    />
                  </View>
                </View>
              )}

              {/* 하단 버튼 */}
              <View style={{ flexDirection: 'row', gap: 12, marginTop: 12 }}>
                {step > 1 && (
                  <Pressable onPress={back} style={[styles.btn, styles.btnGhost]}>
                    <Text style={styles.btnGhostText}>이전</Text>
                  </Pressable>
                )}
                <Pressable
                  onPress={next}
                  disabled={!canProceed}
                  style={[styles.btn, !canProceed && { opacity: 0.5 }]}
                >
                  <Text style={styles.btnText}>{step === totalSteps ? '완료' : '다음'}</Text>
                </Pressable>
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F3F4F6' },
  container: { padding: 16, paddingBottom: 24 },

  title: {
    fontSize: 18,
    color: '#111827',
    letterSpacing: 0.3,
    fontWeight: '300',
  },
  line: { width: 32, height: 1, backgroundColor: '#D1D5DB', alignSelf: 'center', marginTop: 6 },

  progressBar: {
    marginTop: 10,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#E5E7EB',
    overflow: 'hidden',
  },
  progressFill: { height: '100%', backgroundColor: '#111111' },
  progressText: { marginTop: 6, textAlign: 'center', fontSize: 11, color: '#6B7280' },

  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 6 },
    elevation: 2,
  },

  sectionTitle: { fontSize: 16, color: '#111827' },
  sectionSub: { fontSize: 12, color: '#6B7280' },

  input: {
    backgroundColor: '#F9FAFB',
    paddingVertical: 12,
    paddingHorizontal: 12,
    fontSize: 14,
    color: '#111827',
  },

  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },

  selectRow: {
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
    marginTop: 8,
  },
  selectRowActive: { backgroundColor: '#F3F4F6', borderColor: '#D1D5DB' },
  selectLabel: { fontSize: 14, color: '#111827' },

  circle: { width: 18, height: 18, borderRadius: 9, borderWidth: 2 },
  circleSmall: { width: 16, height: 16, borderRadius: 8, borderWidth: 2 },
  circleOn: { backgroundColor: '#111111', borderColor: '#111111' },
  circleOff: { borderColor: '#D1D5DB' },

  grid2: { flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginTop: 8 },

  skinItem: {
    width: '48%',
    padding: 12,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
  },
  skinItemActive: { borderColor: '#111111', backgroundColor: '#F9FAFB' },
  skinRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  skinDot: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D1D5DB',
  },
  skinName: { fontSize: 13, color: '#111827' },

  styleCard: {
    width: '48.5%',
    aspectRatio: 1,
    borderRadius: 8,
    overflow: 'hidden',
    justifyContent: 'flex-end',
    padding: 8,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
  },
  styleCardActive: { borderColor: '#111111', backgroundColor: '#F9FAFB' },
  styleThumb: {
    ...StyleSheet.absoluteFillObject,
  },
  styleName: { color: '#FFF', fontWeight: 'bold', fontSize: 14, textShadowColor: 'rgba(0,0,0,0.5)', textShadowOffset: {width: 0, height: 1}, textShadowRadius: 2 },
  styleOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.3)' },
  styleDesc: { fontSize: 12, color: '#6B7280', lineHeight: 18 },
  grid2: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: 12 },

  tipBox: {
    borderLeftWidth: 2,
    borderLeftColor: '#D1D5DB',
    backgroundColor: '#F9FAFB',
    padding: 12,
    gap: 8,
    marginTop: 8,
  },
  tipTitle: { fontSize: 13, color: '#111827' },
  tipSub: { fontSize: 11, color: '#6B7280' },

  btn: { flex: 1, backgroundColor: '#111111', paddingVertical: 12, alignItems: 'center' },
  btnText: { color: '#FFFFFF', fontSize: 14, letterSpacing: 0.3 },
  btnGhost: { backgroundColor: '#FFFFFF', borderWidth: 1, borderColor: '#E5E7EB' },
  btnGhostText: { color: '#111827', fontSize: 14 },
});
