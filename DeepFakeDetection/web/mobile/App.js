import { useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Image, ScrollView,
  ActivityIndicator, Alert, Platform, SafeAreaView,
  StatusBar as RNStatusBar,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';

// ── Config ─────────────────────────────────────────────────────────────────
// Android emulator → host machine. Change to your LAN IP for a physical device.
// e.g. 'http://192.168.1.100:8000'
const API_BASE = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000';

// ── Design tokens ───────────────────────────────────────────────────────────
const C = {
  bg:          '#0f0f1a',
  glass:       'rgba(255,255,255,0.04)',
  border:      'rgba(255,255,255,0.10)',
  cyan:        '#a2e7ff',
  teal:        '#00e293',
  textPrimary: '#e2e0fc',
  textMuted:   '#c8c5cc',
  red:         '#f87171',
  redBg:       'rgba(239,68,68,0.15)',
  redBorder:   'rgba(239,68,68,0.35)',
  tealBg:      'rgba(0,226,147,0.15)',
  tealBorder:  'rgba(0,226,147,0.35)',
};

function GlassCard({ style, children }) {
  return <View style={[styles.glassCard, style]}>{children}</View>;
}

export default function App() {
  const [image,   setImage]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);

  const pickImage = useCallback(async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Allow photo library access to continue.');
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (!res.canceled) { setImage(res.assets[0]); setResult(null); }
  }, []);

  const takePhoto = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Allow camera access to continue.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (!res.canceled) { setImage(res.assets[0]); setResult(null); }
  }, []);

  const analyze = useCallback(async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);
    try {
      const filename = image.uri.split('/').pop();
      const ext      = filename.split('.').pop().toLowerCase();
      const mime     = ext === 'png' ? 'image/png' : 'image/jpeg';
      const formData = new FormData();
      formData.append('file', { uri: image.uri, name: filename, type: mime });

      const res  = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      Alert.alert('Analysis failed', err.message + '\n\nMake sure the backend is running and reachable.');
    } finally {
      setLoading(false);
    }
  }, [image]);

  const reset = useCallback(() => { setImage(null); setResult(null); }, []);

  const isFake = result?.prediction === 'FAKE';

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <ScrollView style={styles.scroll} contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.badgeRow}>
            <Text style={styles.badgeText}>🛡  ENTERPRISE-GRADE FORENSIC AI</Text>
          </View>
          <Text style={styles.appTitle}>VERIFY_AI</Text>
          <Text style={styles.subtitle}>
            Advanced neural forensic analysis to detect synthetic manipulations in media.
          </Text>
        </View>

        {/* Upload zone */}
        {!result && (
          <GlassCard style={styles.uploadCard}>
            {image ? (
              <Image source={{ uri: image.uri }} style={styles.previewImg} resizeMode="contain" />
            ) : (
              <View style={styles.uploadPrompt}>
                <View style={styles.iconCircle}>
                  <Text style={{ fontSize: 40 }}>🔍</Text>
                </View>
                <Text style={styles.uploadHeadline}>Drop a face image to analyze</Text>
                <Text style={styles.uploadSub}>Tap below to choose an image or take a photo</Text>
              </View>
            )}

            <View style={styles.btnRow}>
              <TouchableOpacity style={styles.outlineBtn} onPress={pickImage} activeOpacity={0.8}>
                <Text style={styles.outlineBtnText}>Gallery</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.outlineBtn} onPress={takePhoto} activeOpacity={0.8}>
                <Text style={styles.outlineBtnText}>Camera</Text>
              </TouchableOpacity>
            </View>

            {image && !loading && (
              <TouchableOpacity style={styles.primaryBtn} onPress={analyze} activeOpacity={0.85}>
                <Text style={styles.primaryBtnText}>ANALYZE FILE</Text>
              </TouchableOpacity>
            )}
            {loading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator size="large" color={C.cyan} />
                <Text style={[styles.muted, { marginTop: 10 }]}>Analyzing…</Text>
              </View>
            )}
          </GlassCard>
        )}

        {/* Results */}
        {result && (
          <>
            <Text style={styles.sectionTitle}>Analysis Results</Text>

            <GlassCard>
              <Text style={styles.cardLabel}>ORIGINAL IMAGE</Text>
              <Image source={{ uri: image.uri }} style={styles.resultImg} resizeMode="contain" />
            </GlassCard>

            <GlassCard>
              <Text style={styles.cardLabel}>AI ATTENTION MAP (GRAD-CAM)</Text>
              <Image
                source={{ uri: `data:image/jpeg;base64,${result.heatmap}` }}
                style={styles.resultImg}
                resizeMode="contain"
              />
            </GlassCard>

            <GlassCard style={styles.verdictCard}>
              <Text style={[styles.cardLabel, { textAlign: 'center' }]}>FORENSIC VERDICT</Text>
              <View style={[
                styles.verdictBadge,
                { backgroundColor: isFake ? C.redBg : C.tealBg, borderColor: isFake ? C.redBorder : C.tealBorder },
              ]}>
                <Text style={{ fontSize: 24 }}>{isFake ? '⚠️' : '✅'}</Text>
                <Text style={[styles.verdictText, { color: isFake ? C.red : C.teal }]}>
                  {result.prediction}
                </Text>
              </View>

              <Text style={[styles.muted, { textAlign: 'center', marginTop: 20, marginBottom: 4 }]}>
                Confidence Score
              </Text>
              <Text style={[styles.confidenceNum, { color: isFake ? C.red : C.teal }]}>
                {result.confidence.toFixed(1)}%
              </Text>

              <View style={styles.barTrack}>
                <View style={[
                  styles.barFill,
                  { width: `${result.confidence}%`, backgroundColor: isFake ? C.red : C.teal },
                ]} />
              </View>

              <Text style={[styles.muted, { textAlign: 'center', marginTop: 14, marginBottom: 20 }]}>
                The model focused on highlighted regions to reach this decision.
              </Text>

              <TouchableOpacity style={[styles.outlineBtn, { alignSelf: 'center', paddingHorizontal: 28 }]} onPress={reset} activeOpacity={0.8}>
                <Text style={styles.outlineBtnText}>Analyze Another Image</Text>
              </TouchableOpacity>
            </GlassCard>
          </>
        )}

        {/* Feature cards */}
        {!result && (
          <View style={{ marginBottom: 4 }}>
            <Text style={styles.sectionTitle}>Multi-Layered Detection Engine</Text>
            <View style={styles.featureGrid}>
              {[
                { icon: '🧬', title: 'Biometric Alignment', body: 'Checks for physiological inconsistencies that AI often fails to replicate in facial rendering.' },
                { icon: '🔬', title: 'Pixel Attribution',   body: 'Tracing source camera metadata and identifying GAN-generated noise patterns at the pixel level.' },
              ].map((f) => (
                <GlassCard key={f.title} style={styles.featureCard}>
                  <Text style={{ fontSize: 28, marginBottom: 10 }}>{f.icon}</Text>
                  <Text style={styles.featureTitle}>{f.title}</Text>
                  <Text style={styles.featureBody}>{f.body}</Text>
                </GlassCard>
              ))}
            </View>
          </View>
        )}

        {/* Protocols */}
        {!result && (
          <GlassCard style={{ marginBottom: 28 }}>
            <Text style={styles.cardLabel}>VERIFICATION PROTOCOLS</Text>
            {['Neural Pattern Recognition', 'Frequency Domain Analysis', 'Artifact Consistency Check'].map((item) => (
              <View key={item} style={styles.protocolRow}>
                <Text style={{ color: C.cyan, fontSize: 16, fontWeight: '700' }}>✓</Text>
                <Text style={styles.protocolText}>{item}</Text>
              </View>
            ))}
          </GlassCard>
        )}

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerBrand}>VERIFY_AI</Text>
          <Text style={styles.footerSub}>© 2024 DEEPFAKE DETECTOR LABS. ALL RIGHTS RESERVED.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:      { flex: 1, backgroundColor: C.bg, paddingTop: Platform.OS === 'android' ? RNStatusBar.currentHeight : 0 },
  scroll:    { flex: 1 },
  container: { padding: 20, paddingBottom: 48 },

  // Header
  header:       { alignItems: 'center', marginBottom: 28, marginTop: 8 },
  badgeRow:     { backgroundColor: 'rgba(162,231,255,0.08)', borderRadius: 20, paddingHorizontal: 14,
                  paddingVertical: 6, borderWidth: 1, borderColor: 'rgba(162,231,255,0.2)', marginBottom: 14 },
  badgeText:    { fontSize: 11, color: C.cyan, letterSpacing: 1.5, fontWeight: '600' },
  appTitle:     { fontSize: 38, fontWeight: '800', color: C.cyan, letterSpacing: -0.5, marginBottom: 10 },
  subtitle:     { fontSize: 15, color: C.textMuted, textAlign: 'center', lineHeight: 22 },

  // Glass card
  glassCard:    { backgroundColor: C.glass, borderWidth: 1, borderColor: C.border, borderRadius: 16, padding: 20, marginBottom: 16 },

  // Upload
  uploadCard:   { marginBottom: 20 },
  uploadPrompt: { alignItems: 'center', paddingVertical: 8 },
  iconCircle:   { width: 80, height: 80, borderRadius: 40, backgroundColor: 'rgba(162,231,255,0.06)',
                  alignItems: 'center', justifyContent: 'center', marginBottom: 14 },
  uploadHeadline: { fontSize: 17, fontWeight: '700', color: C.textPrimary, marginBottom: 6, textAlign: 'center' },
  uploadSub:    { fontSize: 13, color: C.textMuted, textAlign: 'center' },
  previewImg:   { width: '100%', height: 220, borderRadius: 12, marginBottom: 4 },
  btnRow:       { flexDirection: 'row', gap: 12, marginTop: 18 },
  outlineBtn:   { flex: 1, borderWidth: 1, borderColor: 'rgba(162,231,255,0.35)', borderRadius: 8,
                  paddingVertical: 12, alignItems: 'center' },
  outlineBtnText: { fontSize: 12, color: C.cyan, fontWeight: '700', letterSpacing: 1.5, textTransform: 'uppercase' },
  primaryBtn:   { marginTop: 14, backgroundColor: 'rgba(0,210,253,0.9)', borderRadius: 8,
                  paddingVertical: 16, alignItems: 'center' },
  primaryBtnText: { fontSize: 13, fontWeight: '800', letterSpacing: 2, color: '#001f27', textTransform: 'uppercase' },
  loadingRow:   { alignItems: 'center', paddingVertical: 20 },
  muted:        { fontSize: 13, color: C.textMuted },

  // Results
  sectionTitle: { fontSize: 22, fontWeight: '700', color: C.textPrimary, textAlign: 'center', marginBottom: 16 },
  cardLabel:    { fontSize: 11, color: C.textMuted, letterSpacing: 2, fontWeight: '600',
                  textTransform: 'uppercase', marginBottom: 12 },
  resultImg:    { width: '100%', height: 220, borderRadius: 12 },

  // Verdict
  verdictCard:  { alignItems: 'center' },
  verdictBadge: { flexDirection: 'row', alignItems: 'center', gap: 10,
                  paddingHorizontal: 28, paddingVertical: 14, borderRadius: 50, borderWidth: 1, marginTop: 4 },
  verdictText:  { fontSize: 22, fontWeight: '700', letterSpacing: 1 },
  confidenceNum:{ fontSize: 32, fontWeight: '700', textAlign: 'center', marginBottom: 12 },
  barTrack:     { width: '80%', height: 8, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 4, overflow: 'hidden' },
  barFill:      { height: '100%', borderRadius: 4 },

  // Features
  featureGrid:  { flexDirection: 'row', gap: 12 },
  featureCard:  { flex: 1, marginBottom: 16 },
  featureTitle: { fontSize: 14, fontWeight: '700', color: C.textPrimary, marginBottom: 6 },
  featureBody:  { fontSize: 12, color: C.textMuted, lineHeight: 18 },

  // Protocols
  protocolRow:  { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  protocolText: { fontSize: 14, color: C.textPrimary },

  // Footer
  footer:       { alignItems: 'center', paddingTop: 20, borderTopWidth: 1, borderTopColor: C.border },
  footerBrand:  { fontSize: 13, fontWeight: '700', color: C.textPrimary, letterSpacing: 2, marginBottom: 4 },
  footerSub:    { fontSize: 11, color: C.textMuted, textAlign: 'center' },
});
