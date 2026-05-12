import { createRequire } from 'module';
import mammoth from 'mammoth';

const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');

// ── STOPWORDS (NLTK English + resume-specific) ─────────────────────────
const STOPWORDS = new Set([
  'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
  'yourself','yourselves','he','him','his','himself','she','her','hers',
  'herself','it','its','itself','they','them','their','theirs','themselves',
  'what','which','who','whom','this','that','these','those','am','is','are',
  'was','were','be','been','being','have','has','had','having','do','does',
  'did','doing','a','an','the','and','but','if','or','because','as','until',
  'while','of','at','by','for','with','about','against','between','into',
  'through','during','before','after','above','below','to','from','up','down',
  'in','out','on','off','over','under','again','further','then','once',
  'here','there','when','where','why','how','all','both','each','few','more',
  'most','other','some','such','no','nor','not','only','own','same','so',
  'than','too','very','can','will','just','should','now','ll','ve','re',
  'd','s','t','m','ain','aren','couldn','didn','doesn','hadn','hasn',
  'haven','isn','mightn','mustn','needn','shan','shouldn','wasn','weren',
  'won','wouldn',
  // Resume-specific stopwords
  'experience','work','project','projects','using','used','developed','built',
  'responsible','team','company','role','including','also','etc','years',
  'year','month','months','worked','working','good','knowledge','skills',
  'ability','strong','excellent','proficient',
]);

// ── LIGHTWEIGHT PORTER STEMMER ─────────────────────────────────────────
function stem(word) {
  if (word.length < 4) return word;
  // Step 1a
  if (word.endsWith('sses')) return word.slice(0, -2);
  if (word.endsWith('ies') && word.length > 4) return word.slice(0, -2);
  if (word.endsWith('ss')) return word;
  if (word.endsWith('s') && !word.endsWith('ss') && word.length > 4) return word.slice(0, -1);
  // Step 1b
  if (word.endsWith('eed') && word.length > 5) return word.slice(0, -1);
  if (word.endsWith('ing') && word.length > 6) {
    const base = word.slice(0, -3);
    if (base.length >= 3) return base;
  }
  if (word.endsWith('ed') && word.length > 5) {
    const base = word.slice(0, -2);
    if (base.length >= 3) return base;
  }
  // Step 2
  if (word.endsWith('ational') && word.length > 9) return word.slice(0, -5) + 'e';
  if (word.endsWith('izer') && word.length > 6) return word.slice(0, -1);
  if (word.endsWith('ness') && word.length > 6) return word.slice(0, -4);
  if (word.endsWith('ment') && word.length > 6) return word.slice(0, -4);
  if (word.endsWith('ful') && word.length > 6) return word.slice(0, -3);
  if (word.endsWith('ous') && word.length > 6) return word.slice(0, -3);
  if (word.endsWith('ive') && word.length > 6) return word.slice(0, -3);
  if (word.endsWith('ize') && word.length > 6) return word.slice(0, -3);
  if (word.endsWith('ly') && word.length > 5) return word.slice(0, -2);
  if (word.endsWith('er') && word.length > 5) return word.slice(0, -2);
  if (word.endsWith('al') && word.length > 5) return word.slice(0, -2);
  return word;
}

// ── TEXT CLEANING ──────────────────────────────────────────────────────
function cleanText(text) {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/[^\x20-\x7E\n\t]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// ── NLP PREPROCESSING PIPELINE ─────────────────────────────────────────
function preprocess(text) {
  let t = text.toLowerCase();
  t = t.replace(/https?:\/\/\S+|www\.\S+/g, ' ');   // remove URLs
  t = t.replace(/\S+@\S+\.\S+/g, ' ');               // remove emails
  t = t.replace(/\d+/g, ' ');                         // remove digits
  t = t.replace(/[^a-z\s]/g, ' ');                   // keep only a-z

  const tokens = t.split(/\s+/)
    .filter(tok => tok.length > 2 && !STOPWORDS.has(tok));

  return tokens.map(stem).join(' ');
}

// ── TF-IDF TERM COUNTING ───────────────────────────────────────────────
function getTermCounts(preprocessedText) {
  const tokens = preprocessedText.split(/\s+/).filter(Boolean);
  const counts = new Map();

  // Unigrams
  for (const t of tokens) {
    counts.set(t, (counts.get(t) || 0) + 1);
  }
  // Bigrams
  for (let i = 0; i < tokens.length - 1; i++) {
    const bg = `${tokens[i]} ${tokens[i + 1]}`;
    counts.set(bg, (counts.get(bg) || 0) + 1);
  }

  // Sublinear TF: 1 + log(count)
  for (const [t, c] of counts) {
    counts.set(t, 1 + Math.log(c));
  }

  return counts;
}

// ── MATCH SCORE: TF-IDF COSINE (70%) + SKILL OVERLAP (30%) ────────────
function computeMatchScore(resumeText, jdText) {
  const resumeClean = preprocess(resumeText);
  const jdClean = preprocess(jdText);

  const tf1 = getTermCounts(resumeClean);
  const tf2 = getTermCounts(jdClean);

  // Smooth IDF over 2-doc corpus: log((N+1)/(df+1)) + 1
  const allTerms = new Set([...tf1.keys(), ...tf2.keys()]);
  const idf = new Map();
  for (const term of allTerms) {
    const df = (tf1.has(term) ? 1 : 0) + (tf2.has(term) ? 1 : 0);
    idf.set(term, Math.log(3 / (df + 1)) + 1);
  }

  // Cosine similarity
  let dot = 0, norm1 = 0, norm2 = 0;
  for (const term of allTerms) {
    const v1 = (tf1.get(term) || 0) * idf.get(term);
    const v2 = (tf2.get(term) || 0) * idf.get(term);
    dot += v1 * v2;
    norm1 += v1 * v1;
    norm2 += v2 * v2;
  }
  const cosScore = norm1 && norm2 ? dot / (Math.sqrt(norm1) * Math.sqrt(norm2)) : 0;

  // Skill overlap
  const resumeSkillSet = new Set(extractSkills(resumeText));
  const jdSkillSet = new Set(extractSkills(jdText));
  const matchedSkills = [...resumeSkillSet].filter(s => jdSkillSet.has(s));
  const missingSkills = [...jdSkillSet].filter(s => !resumeSkillSet.has(s));
  const extraSkills = [...resumeSkillSet].filter(s => !jdSkillSet.has(s));
  const skillRatio = jdSkillSet.size > 0 ? matchedSkills.length / jdSkillSet.size : 0;

  // Composite score
  const raw = 0.70 * cosScore + 0.30 * skillRatio;
  const finalScore = Math.min(Math.round(raw * 10000) / 100, 100);

  let grade, verdict;
  if (finalScore >= 75) { grade = 'A'; verdict = 'Strong Match — Shortlist'; }
  else if (finalScore >= 55) { grade = 'B'; verdict = 'Good Match — Consider'; }
  else if (finalScore >= 35) { grade = 'C'; verdict = 'Partial Match — Review Manually'; }
  else { grade = 'D'; verdict = 'Weak Match — Not Recommended'; }

  return {
    final_score: finalScore,
    cosine_score: Math.round(cosScore * 10000) / 100,
    skill_score: Math.round(skillRatio * 10000) / 100,
    grade,
    verdict,
    matched_skills: matchedSkills.sort(),
    missing_skills: missingSkills.sort(),
    extra_skills: extraSkills.sort(),
    resume_skills: [...resumeSkillSet].sort(),
    jd_skills: [...jdSkillSet].sort(),
  };
}

// ── SKILL EXTRACTION (keyword matching against known tech skills) ───────
const SKILL_KEYWORDS = [
  // Languages
  'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go',
  'rust', 'kotlin', 'swift', 'scala', 'php', 'ruby', 'r',
  // Web
  'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
  'spring', 'springboot', 'fastapi', 'html', 'css', 'bootstrap',
  // Data / ML
  'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
  'scikit-learn', 'pandas', 'numpy', 'lightgbm', 'xgboost', 'keras',
  'computer vision', 'opencv',
  // Data Engineering
  'sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'redis',
  'kafka', 'spark', 'hadoop', 'hive', 'airflow',
  // Cloud / DevOps
  'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
  'git', 'github', 'ci/cd', 'terraform', 'ansible',
  // Other
  'rest api', 'graphql', 'microservices', 'agile', 'scrum',
  'android', 'ios', 'flutter', 'react native',
];

function extractSkills(text) {
  const tl = text.toLowerCase();
  return [...new Set(SKILL_KEYWORDS.filter(s => tl.includes(s)))].sort();
}

// ── EDUCATION EXTRACTION ───────────────────────────────────────────────
function extractEducation(text) {
  const tl = text.toLowerCase();
  if (['phd', 'doctorate', 'ph.d'].some(k => tl.includes(k))) return 'PhD';
  if (['master', 'm.tech', 'mba', 'msc', 'm.e'].some(k => tl.includes(k))) return 'Masters';
  if (['bachelor', 'b.tech', 'b.e', 'bsc', 'b.sc', 'b.com', 'bca'].some(k => tl.includes(k))) return 'Bachelors';
  if (['diploma', 'polytechnic'].some(k => tl.includes(k))) return 'Diploma';
  return 'Not Detected';
}

// ── EXPERIENCE EXTRACTION ──────────────────────────────────────────────
function extractExperienceYears(text) {
  const tl = text.toLowerCase();
  const patterns = [
    /(\d+\.?\d*)\s*\+?\s*years?\s+of\s+experience/g,
    /(\d+\.?\d*)\s*\+?\s*years?\s+experience/g,
    /experience\s+of\s+(\d+\.?\d*)\s*\+?\s*years?/g,
    /(\d+\.?\d*)\s*\+?\s*yrs?\b/g,
  ];
  const found = [];
  for (const pat of patterns) {
    let m;
    pat.lastIndex = 0;
    while ((m = pat.exec(tl)) !== null) found.push(parseFloat(m[1]));
  }
  return found.length ? Math.max(...found) : 0;
}

// ── FILE TEXT EXTRACTION ───────────────────────────────────────────────
async function extractTextFromFile(buffer, filename) {
  const name = filename.toLowerCase();
  if (name.endsWith('.pdf')) {
    const data = await pdfParse(buffer);
    return data.text;
  }
  if (name.endsWith('.docx') || name.endsWith('.doc')) {
    const result = await mammoth.extractRawText({ buffer });
    return result.value;
  }
  if (name.endsWith('.txt')) {
    return buffer.toString('utf-8');
  }
  throw new Error('Unsupported file type. Upload PDF, DOCX, or TXT.');
}

// ── NETLIFY FUNCTION HANDLER ───────────────────────────────────────────
export default async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 200 });
  }
  if (req.method !== 'POST') {
    return Response.json({ error: 'Method not allowed' }, { status: 405 });
  }

  let formData;
  try {
    formData = await req.formData();
  } catch {
    return Response.json({ error: 'Could not parse form data' }, { status: 400 });
  }

  const file = formData.get('resume');
  const jdText = (formData.get('jd_text') || '').toString().trim();

  if (!file || typeof file === 'string' || file.size === 0) {
    return Response.json({ error: 'No resume file uploaded' }, { status: 400 });
  }
  if (!jdText) {
    return Response.json({ error: 'Job description text is required' }, { status: 400 });
  }

  let resumeText;
  try {
    const buffer = Buffer.from(await file.arrayBuffer());
    resumeText = await extractTextFromFile(buffer, file.name);
  } catch (e) {
    return Response.json({ error: `Text extraction failed: ${e.message}` }, { status: 500 });
  }

  if (!resumeText || !resumeText.trim()) {
    return Response.json({ error: 'Could not extract readable text from the resume file' }, { status: 400 });
  }

  const clean = cleanText(resumeText);
  const skills = extractSkills(clean);
  const education = extractEducation(clean);
  const expYears = extractExperienceYears(clean);
  const scoreData = computeMatchScore(clean, jdText);

  return Response.json({
    status: 'success',
    filename: file.name,
    candidate_info: {
      education,
      experience_years: expYears,
      skills_found: skills,
    },
    ...scoreData,
  });
};

export const config = {
  path: '/api/screen',
};
