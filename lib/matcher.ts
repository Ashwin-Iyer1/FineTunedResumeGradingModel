// Lightweight keyword layer: finds known skills in both texts so each grade
// can explain itself ("You have / They want"). The semantic model handles
// meaning; this layer handles exact asks the model can blur together.

const SKILLS = [
  // Languages
  "python", "javascript", "typescript", "java", "c++", "c#", "c", "go", "rust",
  "swift", "kotlin", "ruby", "php", "scala", "r", "matlab", "sql", "bash",
  // Web & backend
  "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
  "django", "flask", "fastapi", "spring", "graphql", "rest api", "tailwind",
  // Data & ML
  "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
  "pandas", "numpy", "nlp", "computer vision", "data analysis", "statistics",
  "tableau", "power bi", "excel", "spark", "hadoop", "airflow", "etl",
  "data visualization", "llm", "a/b testing",
  // Databases
  "postgresql", "mysql", "mongodb", "redis", "dynamodb", "snowflake", "nosql",
  // Cloud & DevOps
  "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux", "git",
  "ci/cd", "jenkins", "github actions",
  // Engineering
  "solidworks", "autocad", "cad", "fea", "ansys", "labview", "pcb design",
  "embedded systems", "arduino", "plc", "lean", "six sigma", "gd&t",
  // Business & finance
  "financial modeling", "accounting", "valuation", "quickbooks", "sap",
  "salesforce", "crm", "project management", "agile", "scrum", "jira",
  "market research", "seo", "google analytics", "social media",
  "content creation", "adobe creative suite", "figma", "ux research",
  // Science & health
  "pcr", "cell culture", "western blot", "elisa", "hplc", "gel electrophoresis",
  "microscopy", "lab safety", "gmp", "clinical research", "patient care",
  "emr", "cpr", "bls",
  // Operations, sales & people
  "supply chain", "logistics", "inventory management", "procurement",
  "forecasting", "budgeting", "gaap", "cpa", "auditing", "payroll",
  "recruiting", "onboarding", "hris", "workday", "customer service",
  "account management", "business development", "lead generation",
  "negotiation", "cold calling", "hubspot", "microsoft office", "erp",
  "compliance", "hipaa", "osha", "quality assurance", "iso 9001",
  "forklift", "rn", "lpn", "phlebotomy", "medical terminology",
  "curriculum development", "classroom management", "copywriting",
  "editing", "video editing", "photography", "event planning",
  // Communication
  "public speaking", "technical writing", "spanish", "mandarin",
];

const ALIASES: Record<string, string> = {
  js: "javascript",
  ts: "typescript",
  golang: "go",
  "c sharp": "c#",
  nextjs: "next.js",
  "next js": "next.js",
  node: "node.js",
  nodejs: "node.js",
  reactjs: "react",
  "react.js": "react",
  vuejs: "vue",
  postgres: "postgresql",
  mongo: "mongodb",
  k8s: "kubernetes",
  "amazon web services": "aws",
  "google cloud": "gcp",
  "microsoft azure": "azure",
  sklearn: "scikit-learn",
  "scikit learn": "scikit-learn",
  ml: "machine learning",
  "natural language processing": "nlp",
  "large language models": "llm",
  "large language model": "llm",
  llms: "llm",
  "restful": "rest api",
  "rest apis": "rest api",
  "continuous integration": "ci/cd",
  "microsoft excel": "excel",
  powerbi: "power bi",
  "electronic medical records": "emr",
  "electronic health records": "emr",
  ehr: "emr",
  "basic life support": "bls",
  "polymerase chain reaction": "pcr",
  "solid works": "solidworks",
  "finite element analysis": "fea",
  "data analytics": "data analysis",
  "data analyst": "data analysis",
  photoshop: "adobe creative suite",
  illustrator: "adobe creative suite",
  "a/b tests": "a/b testing",
  "ab testing": "a/b testing",
  "registered nurse": "rn",
  "licensed practical nurse": "lpn",
  "certified public accountant": "cpa",
  "ms office": "microsoft office",
  "microsoft 365": "microsoft office",
  "office 365": "microsoft office",
  "talent acquisition": "recruiting",
  "customer support": "customer service",
  "b2b sales": "business development",
  "quality control": "quality assurance",
  "qa/qc": "quality assurance",
  "premiere pro": "video editing",
  "final cut": "video editing",
};

// A specific skill implies the general one a posting may ask for.
const IMPLIES: Record<string, string[]> = {
  pytorch: ["machine learning", "deep learning", "python"],
  tensorflow: ["machine learning", "deep learning", "python"],
  "scikit-learn": ["machine learning", "python"],
  pandas: ["python", "data analysis"],
  numpy: ["python"],
  react: ["javascript"],
  "next.js": ["react", "javascript"],
  angular: ["typescript"],
  vue: ["javascript"],
  "node.js": ["javascript"],
  express: ["node.js", "javascript"],
  django: ["python"],
  flask: ["python"],
  fastapi: ["python"],
  postgresql: ["sql"],
  mysql: ["sql"],
  snowflake: ["sql"],
  mongodb: ["nosql"],
  dynamodb: ["nosql", "aws"],
  kubernetes: ["docker"],
  "github actions": ["ci/cd", "git"],
  jenkins: ["ci/cd"],
  tableau: ["data visualization"],
  "power bi": ["data visualization"],
  scrum: ["agile"],
  solidworks: ["cad"],
  autocad: ["cad"],
  bls: ["cpr"],
  workday: ["hris"],
  hubspot: ["crm"],
  salesforce: ["crm"],
  sap: ["erp"],
  excel: ["microsoft office"],
  cpa: ["accounting", "gaap"],
};

// Short tokens need case/context to avoid matching noise ("go", "rn" in words).
const STRICT = new Set(["c", "r", "go", "rn", "lpn", "erp", "cpa"]);

function escape(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
}

function termRegex(term: string, flags: string) {
  return new RegExp(`(?<![a-z0-9+#.])${escape(term)}(?![a-z0-9+#])`, flags);
}

const SKILL_RES = SKILLS.map((s) => [s, termRegex(s, "i")] as const);
const ALIAS_RES = Object.entries(ALIASES).map(
  ([a, s]) => [s, termRegex(a, "i")] as const
);

export function extractSkills(text: string): Set<string> {
  const found = new Set<string>();
  for (const [skill, re] of SKILL_RES) {
    if (STRICT.has(skill)) {
      // Require the uppercase token in a list-like context, e.g. "C, Java".
      const upper = skill === "go" ? "Go" : skill.toUpperCase();
      if (new RegExp(`(?:^|[\\s,(/;:])${escape(upper)}(?=[\\s,)/;:]|$)`).test(text) &&
          /[,;/]/.test(text)) {
        found.add(skill);
      }
      continue;
    }
    if (re.test(text)) found.add(skill);
  }
  for (const [skill, re] of ALIAS_RES) {
    if (re.test(text)) found.add(skill);
  }
  return found;
}

export type SkillMatch = {
  /** 0-100, share of the posting's skills the resume covers. */
  score: number;
  matched: string[];
  implied: string[];
  missing: string[];
};

export function matchSkills(resumeSkills: Set<string>, jobText: string): SkillMatch {
  const wanted = extractSkills(jobText);
  const impliedHave = new Set<string>();
  for (const s of resumeSkills) {
    for (const g of IMPLIES[s] || []) if (!resumeSkills.has(g)) impliedHave.add(g);
  }

  const matched: string[] = [];
  const implied: string[] = [];
  const missing: string[] = [];
  for (const s of wanted) {
    if (resumeSkills.has(s)) matched.push(s);
    else if (impliedHave.has(s)) implied.push(s);
    else missing.push(s);
  }

  const total = wanted.size;
  // Implied skills count for most, not all, of a direct match. A posting
  // with no recognizable skills is neutral rather than zero.
  const score = total
    ? Math.round(((matched.length + 0.75 * implied.length) / total) * 100)
    : 50;
  return { score, matched, implied, missing };
}
