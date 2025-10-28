// components/common/AppHeader.tsx
import { ArrowLeft } from 'lucide-react-native';
import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';

type AppHeaderProps = {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  rightAction?: React.ReactNode;
};

export default function AppHeader({ title, subtitle, onBack, rightAction }: AppHeaderProps) {
  return (
    <View style={styles.headerWrap}>
      <View style={styles.headerRow}>
        {onBack ? (
          <Pressable onPress={onBack} style={styles.iconBtn}>
            <ArrowLeft size={20} color="#111" />
          </Pressable>
        ) : (
          <View style={styles.iconBtn} />
        )}
        <View style={{ flex: 1, alignItems: 'center' }}>
          <Text style={styles.headerTitle}>{title}</Text>
          {subtitle && <Text style={styles.headerSub}>{subtitle}</Text>}
        </View>
        <View style={styles.rightActionContainer}>
          {rightAction || <View style={styles.iconBtn} />}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  headerWrap: {
    backgroundColor: '#FFFFFF',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E5E7EB',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  iconBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: { fontSize: 18, fontWeight: '600', color: '#0B0B0B' },
  headerSub: { fontSize: 12, color: '#6B7280', marginTop: 2 },
  rightActionContainer: {
    position: 'absolute',
    right: 0,
    top: -2,
  },
});