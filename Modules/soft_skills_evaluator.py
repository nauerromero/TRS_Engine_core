"""
Soft Skills Evaluator - SAORI AI Core V4.0
Evaluates soft skills from text analysis

Key Innovation:
- Complements technical assessment with soft skills
- Keyword-based detection for 5 core soft skills
- Especially important for entry-level candidates
- 20% weight in final score for experienced candidates
- 40% weight for entry-level candidates

Soft Skills Evaluated:
1. Communication
2. Leadership
3. Problem Solving
4. Adaptability
5. Time Management

Usage:
    evaluator = SoftSkillsEvaluator()
    skills = evaluator.evaluate_soft_skills(text)
    overall_score = evaluator.calculate_overall_soft_skills_score(skills)
"""

class SoftSkillsEvaluator:
    """
    Evaluates soft skills based on text analysis (description + responses)
    """
    
    def __init__(self):
        # Keywords para cada soft skill
        self.soft_skills_keywords = {
            "communication": [
                "communicate", "present", "explain", "articulate",
                "collaborate", "team", "meetings", "stakeholders",
                "discuss", "share", "feedback", "listen",
                "write", "document", "report", "presentation"
            ],
            "leadership": [
                "lead", "mentor", "guide", "manage", "coordinate",
                "initiative", "decision", "responsibility", "delegate",
                "motivate", "inspire", "direct", "supervise",
                "organize", "plan", "strategy"
            ],
            "problem_solving": [
                "solve", "optimize", "improve", "analyze", "debug",
                "challenge", "solution", "innovative", "creative",
                "troubleshoot", "resolve", "investigate", "root cause",
                "fix", "identify", "diagnose"
            ],
            "adaptability": [
                "learn", "adapt", "flexible", "change", "growth",
                "new technologies", "fast-paced", "evolving", "dynamic",
                "adjust", "transition", "embrace", "open-minded",
                "agile", "responsive", "versatile"
            ],
            "time_management": [
                "deadline", "prioritize", "organize", "efficient",
                "multitask", "schedule", "planning", "productivity",
                "time", "manage", "balance", "focus",
                "deliver", "on time", "punctual"
            ]
        }
        
        # Pesos por skill (todos iguales por ahora)
        self.skill_weights = {
            "communication": 1.0,
            "leadership": 1.0,
            "problem_solving": 1.0,
            "adaptability": 1.0,
            "time_management": 1.0
        }
    
    def evaluate_soft_skills(self, text):
        """
        Analiza texto y retorna scores de soft skills
        
        Args:
            text: Descripción del candidato o respuestas concatenadas
        
        Returns:
            dict: {skill: score} donde score es 0.0-1.0
        """
        if not text or len(text.strip()) < 10:
            # Texto muy corto, retornar scores neutros
            return {skill: 0.5 for skill in self.soft_skills_keywords.keys()}
        
        text_lower = text.lower()
        scores = {}
        
        for skill, keywords in self.soft_skills_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Normalize to 0.0-1.0
            # Más keywords = mejor score, pero con tope
            max_expected_matches = min(len(keywords), 8)  # Máximo realista
            score = min(matches / max_expected_matches, 1.0)
            
            # Aplicar pesos
            weighted_score = score * self.skill_weights[skill]
            
            scores[skill] = round(weighted_score, 2)
        
        return scores
    
    def calculate_overall_soft_skills_score(self, soft_skills_dict):
        """
        Calcula score general de soft skills
        
        Args:
            soft_skills_dict: dict retornado por evaluate_soft_skills()
        
        Returns:
            float: 0.0-1.0 (promedio de todos los skills)
        """
        if not soft_skills_dict:
            return 0.5  # Neutral si no hay datos
        
        total_score = sum(soft_skills_dict.values())
        count = len(soft_skills_dict)
        
        return round(total_score / count, 2) if count > 0 else 0.5
    
    def get_skill_level(self, score):
        """
        Convierte score numérico a nivel descriptivo
        
        Args:
            score: float (0.0-1.0)
        
        Returns:
            str: Nivel del skill
        """
        if score >= 0.7:
            return "Strong"
        elif score >= 0.4:
            return "Moderate"
        else:
            return "Limited"
    
    def generate_soft_skills_report(self, soft_skills_dict):
        """
        Genera reporte detallado de soft skills
        
        Args:
            soft_skills_dict: dict retornado por evaluate_soft_skills()
        
        Returns:
            str: Reporte formateado
        """
        overall_score = self.calculate_overall_soft_skills_score(soft_skills_dict)
        
        report = "## 🎯 Soft Skills Assessment\n\n"
        report += f"**Overall Score:** {overall_score:.2f} / 1.0\n\n"
        report += "### Individual Skills:\n\n"
        
        # Ordenar por score (descendente)
        sorted_skills = sorted(soft_skills_dict.items(), key=lambda x: x[1], reverse=True)
        
        for skill, score in sorted_skills:
            level = self.get_skill_level(score)
            emoji = "🟢" if level == "Strong" else ("🟡" if level == "Moderate" else "🔴")
            
            skill_name = skill.replace("_", " ").title()
            report += f"{emoji} **{skill_name}:** {score:.2f} ({level})\n"
        
        # Recomendaciones
        report += "\n### 💡 Recommendations:\n\n"
        
        weak_skills = [skill for skill, score in soft_skills_dict.items() if score < 0.4]
        strong_skills = [skill for skill, score in soft_skills_dict.items() if score >= 0.7]
        
        if strong_skills:
            strong_names = [s.replace("_", " ").title() for s in strong_skills]
            report += f"✅ **Strengths:** {', '.join(strong_names)}\n"
        
        if weak_skills:
            weak_names = [s.replace("_", " ").title() for s in weak_skills]
            report += f"⚠️ **Areas for Improvement:** {', '.join(weak_names)}\n"
        
        if not weak_skills and not strong_skills:
            report += "🟡 Candidate demonstrates moderate soft skills across all areas.\n"
        
        return report
    
    def detect_soft_skill_strengths(self, soft_skills_dict, threshold=0.7):
        """
        Identifica los soft skills fuertes del candidato
        
        Args:
            soft_skills_dict: dict retornado por evaluate_soft_skills()
            threshold: Score mínimo para considerar "fuerte"
        
        Returns:
            list: Lista de skills fuertes
        """
        return [
            skill.replace("_", " ").title() 
            for skill, score in soft_skills_dict.items() 
            if score >= threshold
        ]
    
    def detect_soft_skill_gaps(self, soft_skills_dict, threshold=0.4):
        """
        Identifica gaps en soft skills
        
        Args:
            soft_skills_dict: dict retornado por evaluate_soft_skills()
            threshold: Score máximo para considerar "gap"
        
        Returns:
            list: Lista de skills con gap
        """
        return [
            skill.replace("_", " ").title() 
            for skill, score in soft_skills_dict.items() 
            if score < threshold
        ]


def evaluate_entry_level_candidate(profile, soft_skills_score):
    """
    Evaluación especial para candidatos entry-level (0-2 años experiencia)
    
    LÓGICA CONDICIONAL (v2.0):
    Aplica ajuste SOLO cuando es beneficioso:
    - Enthusiastic: Siempre (bonus 0.15 compensa)
    - High soft skills (>=0.10): Siempre
    - Medium soft skills (>=0.08): Solo si match bajo (<30%)
    - Otros casos: NO (sería contraproducente)
    
    Args:
        profile: dict con datos del candidato
        soft_skills_score: float (0.0-1.0)
    
    Returns:
        dict: {
            "match_weight": float,
            "soft_skills_weight": float,
            "enthusiasm_bonus": float,
            "education_bonus": float,
            "total_bonus": float,
            "adjustment_applied": bool,
            "adjustment_reason": str
        }
    """
    experience_years = profile.get("experience_years", 0)
    
    # Solo aplicar para entry-level (actualizado a 2 años)
    if experience_years > 2:
        return {
            "match_weight": 1.0,
            "soft_skills_weight": 0.2,
            "enthusiasm_bonus": 0.0,
            "education_bonus": 0.0,
            "total_bonus": 0.0,
            "adjustment_applied": False,
            "adjustment_reason": "Not entry-level (>2 years)"
        }
    
    # Es entry-level (0-2 años)
    # Decidir si aplicar adjustment basado en criterios inteligentes
    
    emotional_state = profile.get("emotional_state", "").lower()
    is_enthusiastic = emotional_state in ["enthusiastic", "confident", "positive"]
    
    # CRITERIOS DE APLICACIÓN:
    # 1. Enthusiastic: Siempre aplicar (bonus 0.15 compensa reducción de match)
    # 2. High soft skills (>=0.10): Aplicar (soft skills compensan)
    # 3. Medium soft skills (>=0.08): Aplicar (asumimos low match en entry-level)
    # 4. Otros: NO aplicar (sería contraproducente)
    
    should_apply_adjustment = (
        is_enthusiastic or
        soft_skills_score >= 0.10 or
        soft_skills_score >= 0.08
    )
    
    if should_apply_adjustment:
        # Aplicar ajuste entry-level
        match_weight = 0.6  # Reduce importancia del match técnico
        soft_skills_weight = 0.4  # Aumenta importancia de soft skills
        enthusiasm_bonus = 0.15 if is_enthusiastic else 0.0
        
        if is_enthusiastic:
            reason = "Enthusiastic state (bonus compensates)"
        elif soft_skills_score >= 0.10:
            reason = f"High soft skills ({soft_skills_score:.2f})"
        else:
            reason = f"Medium soft skills ({soft_skills_score:.2f})"
        
        adjustment_applied = True
    else:
        # NO aplicar ajuste (mantener pesos normales)
        match_weight = 1.0
        soft_skills_weight = 0.2
        enthusiasm_bonus = 0.0
        reason = f"Low soft skills ({soft_skills_score:.2f}) - standard weights better"
        adjustment_applied = False
    
    # Bonus por educación relevante
    has_education = profile.get("education") or profile.get("certifications")
    education_bonus = 0.10 if has_education else 0.0
    
    # Bonus adicional si soft skills son fuertes
    soft_skills_bonus = 0.05 if soft_skills_score >= 0.7 else 0.0
    
    total_bonus = enthusiasm_bonus + education_bonus + soft_skills_bonus
    
    return {
        "match_weight": match_weight,
        "soft_skills_weight": soft_skills_weight,
        "enthusiasm_bonus": enthusiasm_bonus,
        "education_bonus": education_bonus,
        "soft_skills_bonus": soft_skills_bonus,
        "total_bonus": round(total_bonus, 2),
        "adjustment_applied": adjustment_applied,
        "adjustment_reason": reason
    }


# Example usage and testing
if __name__ == "__main__":
    evaluator = SoftSkillsEvaluator()
    
    print("=" * 80)
    print("SOFT SKILLS EVALUATOR - TEST CASES")
    print("=" * 80)
    
    # Test Case 1: Candidate with strong communication and leadership
    print("\n📊 TEST 1: Leadership-focused candidate")
    print("-" * 80)
    
    text1 = """
    I'm very excited about this opportunity to lead and mentor a team. Throughout my 
    career, I've consistently taken initiative to organize projects, coordinate with 
    stakeholders, and guide junior developers. I pride myself on my ability to 
    communicate complex technical concepts clearly to both technical and non-technical 
    audiences. I regularly present project updates, facilitate team meetings, and 
    collaborate with cross-functional teams to deliver innovative solutions.
    """
    
    skills1 = evaluator.evaluate_soft_skills(text1)
    overall1 = evaluator.calculate_overall_soft_skills_score(skills1)
    
    print(f"Overall Soft Skills Score: {overall1:.2f}\n")
    for skill, score in skills1.items():
        level = evaluator.get_skill_level(score)
        print(f"  {skill.replace('_', ' ').title():20} {score:.2f} ({level})")
    
    strengths1 = evaluator.detect_soft_skill_strengths(skills1)
    if strengths1:
        print(f"\n✅ Strengths: {', '.join(strengths1)}")
    
    # Test Case 2: Technical candidate with limited soft skills
    print("\n\n📊 TEST 2: Technical-focused candidate")
    print("-" * 80)
    
    text2 = """
    I have experience with Python and databases. I can code and implement solutions.
    I work on projects and deliver results.
    """
    
    skills2 = evaluator.evaluate_soft_skills(text2)
    overall2 = evaluator.calculate_overall_soft_skills_score(skills2)
    
    print(f"Overall Soft Skills Score: {overall2:.2f}\n")
    for skill, score in skills2.items():
        level = evaluator.get_skill_level(score)
        print(f"  {skill.replace('_', ' ').title():20} {score:.2f} ({level})")
    
    gaps2 = evaluator.detect_soft_skill_gaps(skills2)
    if gaps2:
        print(f"\n⚠️ Development Areas: {', '.join(gaps2)}")
    
    # Test Case 3: Entry-level candidate evaluation
    print("\n\n📊 TEST 3: Entry-level candidate adjustment")
    print("-" * 80)
    
    profile_entry = {
        "name": "Junior Developer",
        "experience_years": 0,
        "emotional_state": "enthusiastic",
        "education": "Computer Science Degree",
        "certifications": ["AWS Cloud Practitioner"]
    }
    
    text3 = """
    I'm eager to learn and adapt to new technologies. I prioritize my tasks and 
    manage my time efficiently to meet deadlines. I'm flexible and open to feedback, 
    always looking to improve my skills through continuous learning.
    """
    
    skills3 = evaluator.evaluate_soft_skills(text3)
    overall3 = evaluator.calculate_overall_soft_skills_score(skills3)
    adjustments = evaluate_entry_level_candidate(profile_entry, overall3)
    
    print(f"Soft Skills Score: {overall3:.2f}")
    print(f"\nEntry-Level Adjustments:")
    print(f"  Match Weight: {adjustments['match_weight']} (reduced from 1.0)")
    print(f"  Soft Skills Weight: {adjustments['soft_skills_weight']} (increased from 0.2)")
    print(f"  Enthusiasm Bonus: +{adjustments['enthusiasm_bonus']}")
    print(f"  Education Bonus: +{adjustments['education_bonus']}")
    print(f"  Total Bonus: +{adjustments['total_bonus']}")
    
    # Test Case 4: Full report generation
    print("\n\n📊 TEST 4: Full Soft Skills Report")
    print("-" * 80)
    
    report = evaluator.generate_soft_skills_report(skills1)
    print(report)
    
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)

