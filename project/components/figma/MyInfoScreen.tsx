// components/figma/MyInfoScreen.tsx
import React, { useEffect, useState } from 'react';
import {
  View, Text, Image, StyleSheet, Pressable, TextInput,
  Alert, ActivityIndicator
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ChevronLeft } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';

// ✅ 로그인과 동일한 도메인으로 통일 (필요시 한 곳만 바꾸면 됨)
const API_BASE = 'https://loyd-extemporaneous-annalise.ngrok-free.dev';

type RawUser = {
  user_id?: number | string;
  id?: number | string;
  name: string;
  email: string;
  avatar?: string;
  bio?: string;
};
type User = {
  id: string;                 // 표준화된 id(문자열)
  name: string;
  email: string;
  avatar?: string;
  bio?: string;
};
type Props = { onBack?: () => void };

/** 서버/스토리지의 다양한 형태를 표준 User로 변환 */
function normalizeUser(u: RawUser): User {
  const id = u.id ?? u.user_id;      // user_id 우선 반영
  return {
    id: String(id ?? ''),            // 빈값 방지
    name: u.name,
    email: u.email,
    avatar: u.avatar,
    bio: u.bio,
  };
}

export default function MyInfoScreen({ onBack }: Props) {
  const [user, setUser] = useState<User | null>(null);
  const [name, setName] = useState('');
  const [bio, setBio] = useState('');
  const [avatar, setAvatar] = useState<string | undefined>(undefined);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      // 1순위: @kko/user, 2순위: user(LoginScreen에서 저장)
      const raw1 = await AsyncStorage.getItem('@kko/user');
      const raw2 = !raw1 ? await AsyncStorage.getItem('user') : null;
      const raw = raw1 ?? raw2;
      if (!raw) return;

      try {
        const parsed: RawUser = JSON.parse(raw);
        const nu = normalizeUser(parsed);
        setUser(nu);
        setName(nu.name ?? '');
        setBio(nu.bio ?? '');
        setAvatar(nu.avatar);
        // 표준 키로 재저장(다음부터는 @kko/user만 사용)
        await AsyncStorage.setItem('@kko/user', JSON.stringify(nu));
      } catch (e) {
        console.warn('user parse failed', e);
      }
    })();
  }, []);

  const pickAvatar = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') return Alert.alert('사진 접근 권한이 필요합니다.');
    const r = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (!r.canceled) setAvatar(r.assets[0].uri);
  };

  const saveProfile = async () => {
    if (!user) return;
    setBusy(true);
    try {
      const token = await AsyncStorage.getItem('@kko/token'); // 있을 때만 사용
      // ⚠️ 서버 스펙: PUT /api/users/:id 가정
      const res = await fetch(`${API_BASE}/api/users/${user.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ name, bio, avatar }),
      });
      if (!res.ok) throw new Error('update_failed');

      const updated = { ...user, name, bio, avatar };
      await AsyncStorage.setItem('@kko/user', JSON.stringify(updated));
      setUser(updated);
      Alert.alert('저장 완료', '프로필이 업데이트되었어요.');
    } catch {
      Alert.alert('오류', '프로필 저장에 실패했습니다.');
    } finally {
      setBusy(false);
    }
  };

  const logout = async () => {
    await AsyncStorage.multiRemove(['@kko/token', '@kko/user', 'user']);
    Alert.alert('로그아웃', '다시 로그인 화면으로 이동하세요.');
  };

  const deleteAccount = async () => {
    if (!user) return;
    Alert.alert('회원 탈퇴', '정말로 탈퇴하시겠어요? (되돌릴 수 없음)', [
      { text: '취소', style: 'cancel' },
      {
        text: '탈퇴',
        style: 'destructive',
        onPress: async () => {
          try {
            setBusy(true);
            const token = await AsyncStorage.getItem('@kko/token');
            // ⚠️ 서버 스펙: DELETE /api/users/:id (me만 받으면 여기만 바꿔서 /api/users/me 로)
            const res = await fetch(`${API_BASE}/api/users/${user.id}`, {
              method: 'DELETE',
              headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
            });

            const text = await res.text(); // 디버깅 도움
            console.log('DELETE /users/:id ->', res.status, text);

            if (!res.ok) {
              Alert.alert('오류', `탈퇴 처리에 실패했습니다.\nstatus: ${res.status}\n${text}`);
              return;
            }

            await AsyncStorage.multiRemove(['@kko/token', '@kko/user', 'user']);
            Alert.alert('탈퇴 완료', '그동안 이용해주셔서 감사합니다.');
          } catch (e: any) {
            Alert.alert('오류', `탈퇴 처리 중 예외가 발생했습니다.\n${String(e?.message ?? e)}`);
          } finally {
            setBusy(false);
          }
        },
      },
    ]);
  };

  if (!user) {
    return (
      <View style={S.center}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <View style={S.wrap}>
      {/* 상단 안전영역 + 뒤로가기(아이폰에서 터치 잘 되도록) */}
      {onBack && (
        <SafeAreaView edges={['top']} style={{ backgroundColor: '#fff' }}>
          <View style={H.bar}>
            <Pressable onPress={onBack} hitSlop={12} style={H.back}>
              <ChevronLeft size={24} color="#111" />
            </Pressable>
            <Text style={H.title}>내 정보</Text>
            <View style={{ width: 36 }} />
          </View>
        </SafeAreaView>
      )}

      <Pressable style={S.avatarWrap} onPress={pickAvatar}>
        {avatar ? (
          <Image source={{ uri: avatar }} style={S.avatar} />
        ) : (
          <View style={[S.avatar, S.avatarEmpty]}><Text>사진</Text></View>
        )}
        <Text style={S.small}>프로필 사진 변경</Text>
      </Pressable>

      <View style={S.card}>
        <Label>이메일</Label>
        <Text style={S.value}>{user.email}</Text>

        <Label>닉네임</Label>
        <TextInput
          style={S.input}
          value={name}
          onChangeText={setName}
          placeholder="닉네임"
        />

        <Label>소개</Label>
        <TextInput
          style={[S.input, { height: 90 }]}
          value={bio}
          onChangeText={setBio}
          placeholder="한 줄 소개"
          multiline
        />

        <Primary
          onPress={saveProfile}
          disabled={busy}
          label={busy ? '저장 중...' : '변경사항 저장'}
        />
      </View>

      <View style={[S.card, { gap: 12 }]}>
        <Secondary onPress={logout} label="로그아웃" />
        <Danger onPress={deleteAccount} label="회원 탈퇴" />
      </View>
    </View>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <Text style={S.label}>{children}</Text>;
}
function Primary({ onPress, label, disabled }: any) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={[S.btn, S.btnPrimary, disabled && { opacity: 0.7 }]}
    >
      <Text style={S.btnTextPrimary}>{label}</Text>
    </Pressable>
  );
}
function Secondary({ onPress, label }: any) {
  return (
    <Pressable onPress={onPress} style={[S.btn, S.btnSecondary]}>
      <Text style={S.btnTextSecondary}>{label}</Text>
    </Pressable>
  );
}
function Danger({ onPress, label }: any) {
  return (
    <Pressable onPress={onPress} style={[S.btn, S.btnDanger]}>
      <Text style={S.btnTextDanger}>{label}</Text>
    </Pressable>
  );
}

const H = StyleSheet.create({
  bar: {
    height: 56,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#eee',
  },
  back: { padding: 6 },
  title: { flex: 1, textAlign: 'center', fontSize: 18, fontWeight: '700', color: '#111' },
});

const S = StyleSheet.create({
  wrap: { flex: 1, padding: 20, backgroundColor: '#fafafa', gap: 16 },
  avatarWrap: { alignItems: 'center', gap: 6, marginTop: 4 },
  avatar: { width: 96, height: 96, borderRadius: 48 },
  avatarEmpty: { backgroundColor: '#eee', alignItems: 'center', justifyContent: 'center' },
  small: { color: '#777', fontSize: 12 },
  card: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 14,
    gap: 8,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 6,
    elevation: 1,
  },
  label: { color: '#666', marginTop: 8, marginBottom: 2 },
  value: { fontSize: 16, fontWeight: '600' },
  input: { borderWidth: 1, borderColor: '#e6e6e6', borderRadius: 12, padding: 12, backgroundColor: '#fff' },
  btn: { paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  btnPrimary: { backgroundColor: '#191919' }, btnTextPrimary: { color: '#fff', fontWeight: '700' },
  btnSecondary: { backgroundColor: '#f3f3f3' }, btnTextSecondary: { color: '#333', fontWeight: '600' },
  btnDanger: { backgroundColor: '#ffe9e9' }, btnTextDanger: { color: '#d21f3c', fontWeight: '700' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
});
