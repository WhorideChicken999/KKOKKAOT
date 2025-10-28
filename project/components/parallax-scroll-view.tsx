import type { PropsWithChildren, ReactElement } from 'react';
import React from 'react';
import { StyleSheet } from 'react-native';
import Animated, {
  interpolate,
  useAnimatedRef, // useAnimatedRef는 여전히 스크롤뷰 참조를 위해 필요할 수 있습니다.
  useAnimatedStyle,
  useSharedValue, // [변경] useScrollOffset 대신 사용
  useAnimatedScrollHandler, // [변경] useScrollOffset 대신 사용
} from 'react-native-reanimated';

import { ThemedView } from '@/components/themed-view';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { useThemeColor } from '@/hooks/use-theme-color';

const HEADER_HEIGHT = 250;

type Props = PropsWithChildren<{
  headerImage: ReactElement;
  headerBackgroundColor: { dark: string; light: string };
}>;

const ParallaxScrollView = React.forwardRef<Animated.ScrollView, Props>(
  ({ children, headerImage, headerBackgroundColor }, ref) => {
    const backgroundColor = useThemeColor({}, 'background');
    const colorScheme = useColorScheme() ?? 'light';

    // [변경] useScrollOffset을 useSharedValue와 useAnimatedScrollHandler로 대체합니다.
    const scrollOffset = useSharedValue(0);
    const scrollHandler = useAnimatedScrollHandler({
      onScroll: (event) => {
        scrollOffset.value = event.contentOffset.y;
      },
    });

    const headerAnimatedStyle = useAnimatedStyle(() => {
      return {
        transform: [
          {
            translateY: interpolate(
              scrollOffset.value, // .value로 현재 스크롤 위치에 접근합니다.
              [-HEADER_HEIGHT, 0, HEADER_HEIGHT],
              [-HEADER_HEIGHT / 2, 0, HEADER_HEIGHT * 0.75]
            ),
          },
          {
            scale: interpolate(
              scrollOffset.value,
              [-HEADER_HEIGHT, 0, HEADER_HEIGHT],
              [2, 1, 1]
            ),
          },
        ],
      };
    });

    return (
      <Animated.ScrollView
        // [수정] 이제 부모로부터 받은 ref를 여기에 바로 연결해도 타입 문제가 없습니다.
        ref={ref}
        style={{ backgroundColor, flex: 1 }}
        scrollEventThrottle={16}
        // [추가] 생성된 스크롤 핸들러를 onScroll 이벤트에 연결합니다.
        onScroll={scrollHandler}>
        <Animated.View
          style={[
            styles.header,
            { backgroundColor: headerBackgroundColor[colorScheme] },
            headerAnimatedStyle,
          ]}>
          {headerImage}
        </Animated.View>
        <ThemedView style={styles.content}>{children}</ThemedView>
      </Animated.ScrollView>
    );
  }
);

export default ParallaxScrollView;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    height: HEADER_HEIGHT,
    overflow: 'hidden',
  },
  content: {
    flex: 1,
    padding: 32,
    gap: 16,
    overflow: 'hidden',
  },
});