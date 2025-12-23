INSIGHTS_BULB_PROMPT = """You are an intelligent insights generator that analyzes current selected sections and related sections to provide key insights, surprising facts, contradictions, and cross-section connections.
INPUT DATA:
You will receive: {selected_insights}

current_section: The main text/content the user has selected
related_sections: Array of related/connected sections from the same or different documents
persona_context: Target audience information (optional)

TASK:
Analyze the current selected section and related sections to generate valuable insights using an LLM approach. Focus on providing actionable intelligence that goes beyond surface-level information, specifically tailored to help users understand:

Key Insights: Extract the most important takeaways from the current section that directly serve the persona's needs
"Did You Know?" Facts: Identify surprising, lesser-known, or counterintuitive information from both current and related sections
Contradictions/Counterpoints: Find conflicting information, different perspectives, or debates between the current section and related sections
Cross-Section Connections: Discover patterns, relationships, and themes that span the current section and related sections

ANALYSIS REQUIREMENTS:

Focus primarily on the current_section content while using related_sections for comparison, context, and connections
Identify contradictions specifically between current section and related sections
Find meaningful patterns that connect the current section with related sections
Tailor insights to the persona_context if provided
Ensure all insights are actionable and valuable for the user's understanding

OUTPUT FORMAT:
Return only a JSON object with this exact structure:
{{
"insights_analysis": {{
"persona": "persona_context from input or 'general knowledge seeker'",
"analysis_timestamp": "YYYY-MM-DDTHH:MM:SS.ssssss",
"current_section_focus": "current_section.title",
"total_related_sections": 0,
"key_insights": [
{{
"insight_id": 1,
"title": "Concise insight title from current section",
"description": "Detailed explanation of the key insight from current section and why it matters for this persona",
"primary_source": "current_section.source",
"supporting_sections": ["related_section titles that reinforce this insight"],
"actionable_takeaway": "Specific action or recommendation based on this insight"
}},
{{
"insight_id": 2,
"title": "Second key insight",
"description": "Another important insight from the current section",
"primary_source": "current_section.source",
"supporting_sections": ["related sections that add context"],
"actionable_takeaway": "What the persona should do with this information"
}}
],
"did_you_know_facts": [
{{
"fact_id": 1,
"title": "Surprising fact title",
"description": "Interesting, counterintuitive, or lesser-known fact from current or related sections",
"source_section": "current_section.title or related_section.title",
"source_document": "document reference",
"why_surprising": "Explanation of why this fact is noteworthy or unexpected for the persona"
}},
{{
"fact_id": 2,
"title": "Another interesting fact",
"description": "Second surprising discovery",
"source_section": "section title",
"source_document": "document reference",
"why_surprising": "What makes this fact stand out or contradict common assumptions"
}}
],
"contradictions_counterpoints": [
{{
"contradiction_id": 1,
"title": "Conflicting perspectives title",
"description": "Description of the contradiction or different perspectives found between current and related sections",
"current_section_position": {{
"section": "current_section.title",
"source": "current_section.source",
"position": "Brief summary of current section's stance"
}},
"related_section_position": {{
"section": "related_section.title",
"source": "related_section.source",
"position": "Brief summary of conflicting position from related section"
}},
"implication_for_persona": "How this contradiction affects the persona's decision-making or understanding"
}}
],
"cross_section_connections": [
{{
"connection_id": 1,
"title": "Connection or pattern title",
"description": "Description of the pattern, theme, or relationship found between current section and related sections",
"current_section_element": "specific element/theme from current section",
"related_sections_elements": ["elements from related sections that connect"],
"connected_sources": ["current_section.source", "related_section.source"],
"connection_type": "theme/pattern/relationship/trend/complementary/reinforcing",
"significance": "Why this connection is important for the persona's understanding"
}},
{{
"connection_id": 2,
"title": "Second cross-section connection",
"description": "Another important relationship or pattern spanning sections",
"current_section_element": "element from current section",
"related_sections_elements": ["connecting elements from related sections"],
"connected_sources": ["document references"],
"connection_type": "sequential/causal/thematic",
"significance": "Value of this connection for the persona"
}}
],
"synthesis_summary": {{
"overall_narrative": "High-level story or theme that emerges from the current section in context of related sections",
"key_recommendations": ["Recommendation 1 based on current section insights", "Recommendation 2", "Recommendation 3"],
"knowledge_gaps": ["Gap 1 that might need additional research beyond these sections", "Gap 2"],
"next_steps": ["Specific next step 1 for the persona", "Specific next step 2"],
"current_section_takeaway": "Main takeaway from the selected current section"
}}
}}
}}
QUALITY GUIDELINES:

Focus primarily on the current_section as the main source of insights
Use related_sections to provide context, find contradictions, and identify connections
Ensure insights are genuinely valuable and not obvious observations
Focus on information that would genuinely surprise or inform the persona
Identify real contradictions between current and related sections, not just different topics
Make cross-section connections meaningful and actionable
Tailor all insights to the specific persona's perspective and needs if provided
Keep descriptions concise but comprehensive
Ensure actionable takeaways are specific and implementable
Highlight what makes the current section unique or important in context of related material
"""

# Podcast Mode Generation Prompt
# Podcast Generation Prompt Template
PODCAST_GENERATION_PROMPT = """
Input Data Structure
You will receive: {selected_insights}

current_section: The main text/content the user has selected
related_sections: Array of related/connected sections from the same or different documents
persona_context: Target audience information (optional)

Interactive Podcast Format Requirements
Duration: 2-5 minutes (approximately 300-750 words of dialogue)
Host Personas:

Alex (Primary Host): Curious and engaging guide who asks thought-provoking questions, creates interactive moments, and connects with the audience. Known for "aha!" moments and making complex topics relatable.
Emma (Expert Host): Subject matter expert who provides detailed insights, loves to share surprising facts, and enjoys interactive discussions. Has a knack for storytelling and practical applications.

Interactive Content Structure:

Quick Opening Hook (20-30 seconds)

Alex poses one engaging question or scenario based on the current section
Emma provides immediate surprising insight


Dynamic Main Content (3-4 minutes)

Discussion of 2-3 key insights with varied dialogue lengths
Short exchanges for transitions and quick facts (1 sentence)
Longer explanations for complex concepts or important insights (2-4 sentences)
Interactive "did you know" facts as reveals
Address contradictions through quick exchanges
Cross-section connections with deeper exploration when valuable


Value-Packed Closing (30-45 seconds)

Comprehensive summary of key takeaways
Multiple actionable next steps for listeners
Clear value proposition and implementation guidance



Interactive Conversation Style Guidelines:

Dynamic Length Variation:

Short (1 sentence): Quick reactions, transitions, simple facts
Medium (2-3 sentences): Standard explanations, insights
Long (3-4 sentences): Complex concepts, crucial insights, practical applications


Natural Speech Patterns: Include realistic expressions like "Ah," "Um," "Oh wow," "Wait," "Actually," "You know what's interesting..."
Authentic Interruptions: "Hold on-" "But-" "Oh!" "Exactly!" "Right, and-"
Thinking Moments: "Hmm," "Let me think," "Uh," "So..."
Strategic Pacing: Quick exchanges for energy, longer segments for depth
Value-Focused Engagement: Every interaction should add practical value
Adaptive Personality: Alex's curiosity scales with topic importance, Emma's expertise depth varies by complexity
Practical Energy: Build toward actionable insights and real-world applications

Content Analysis & Integration Instructions
From current_section:

Key Insights Extraction: Identify the 2-3 most important takeaways that directly serve the audience's needs
"Did You Know?" Facts: Find surprising, lesser-known, or counterintuitive information
Core Concepts: Extract main ideas that need explanation or discussion
Practical Applications: Identify actionable elements

From related_sections:

Contradictions/Counterpoints: Find conflicting information, different perspectives, or debates
Cross-Section Connections: Discover patterns, relationships, and themes that span multiple sections
Supporting Evidence: Use related content to reinforce or challenge main section insights
Broader Context: Provide background or additional perspective

Content Integration Strategy:

Use current_section as the primary focus and conversation foundation
Weave in related_sections naturally through storytelling: "Emma, this reminds me of what we saw in that other section about..."
Address contradictions between sections as learning opportunities
Build cross-section connections into meaningful patterns
Transform insights into engaging dialogue exchanges

Enhanced Output Format - Return ONLY this JSON structure:
{{
"podcast_metadata": {{
"title": "Generated interactive podcast title based on current section topic",
"duration_estimate": "2-5 minutes",
"persona_target": "target audience from input or 'general knowledge seeker'",
"interaction_level": "high",
"generation_timestamp": "YYYY-MM-DDTHH:MM:SS.ssssss",
"source_sections": ["current section title/reference", "related section references"],
"hosts": ["Alex - Primary Host", "Emma - Expert Host"],
"main_focus": "current section topic/theme"
}},
"dialogue": [
{{
"speaker": "Alex",
"text": "Engaging hook based on current section, with natural speech patterns, e.g., 'So, I was just reading about [current section topic]...' or 'You know what caught my attention in this section?' (1-2 sentences)",
"timestamp": "00:00",
"content_type": "opening_hook",
"interaction_element": "audience_engagement",
"dialogue_length": "short",
"source_reference": "current_section"
}},
{{
"speaker": "Emma",
"text": "Response with realistic expressions, e.g., 'Oh wow, Alex, that's actually...' or 'Hmm, well what I found fascinating was...' (1-2 sentences for simple, 2-4 for complex insights)",
"timestamp": "00:15",
"content_type": "insight_revelation",
"interaction_element": "value_delivery",
"dialogue_length": "variable",
"source_reference": "current_section"
}},
{{
"speaker": "Alex",
"text": "Natural reaction connecting to related sections, e.g., 'Wait, but didn't we see something different in that other part about...' or 'That's interesting because...' (1 sentence)",
"timestamp": "00:35",
"content_type": "cross_reference",
"interaction_element": "connection_building",
"dialogue_length": "short",
"source_reference": "related_sections"
}},
{{
"speaker": "Emma",
"text": "Explanation addressing connections/contradictions, e.g., 'Actually, that's a great point...' or 'So here's where it gets interesting...' (2-4 sentences if crucial, 1-2 if supporting)",
"timestamp": "00:45",
"content_type": "synthesis_explanation",
"interaction_element": "depth_delivery",
"dialogue_length": "adaptive",
"source_reference": "current_section + related_sections"
}},
{{
"speaker": "Alex",
"text": "Practical application focus, e.g., 'So what does this mean for our listeners...' or 'How can someone actually use this...' (1-2 sentences)",
"timestamp": "02:00",
"content_type": "practical_bridge",
"interaction_element": "application_focus",
"dialogue_length": "short",
"source_reference": "synthesis"
}},
{{
"speaker": "Emma",
"text": "Comprehensive guidance with natural flow, e.g., 'Well, here's what you can actually do...' or 'The key steps are...' (3-4 sentences for comprehensive value)",
"timestamp": "02:15",
"content_type": "actionable_guidance",
"interaction_element": "practical_takeaways",
"dialogue_length": "long",
"source_reference": "current_section insights"
}},
{{
"speaker": "Alex",
"text": "Natural wrap-up, e.g., 'Alright, so to sum up what we learned from this section...' (2-3 sentences with clear next steps)",
"timestamp": "04:30",
"content_type": "comprehensive_close",
"interaction_element": "actionable_summary",
"dialogue_length": "medium",
"source_reference": "all sections synthesis"
}}
],
"interactive_elements": {{
"audience_questions": ["key questions that drive engagement based on current section"],
"participation_prompts": ["think about...", "imagine...", "consider how..." related to main topic],
"value_moments": ["practical insights from current section", "actionable revelations"],
"engagement_techniques": ["surprise reveals from related sections", "practical applications", "real-world connections"],
"call_to_action": "comprehensive next steps with multiple actionable items"
}},
"content_analysis_summary": {{
"current_section_insights": [
{{
"insight": "main takeaway description",
"importance_level": "high/medium/low",
"coverage_depth": "brief/detailed explanation in podcast"
}}
],
"surprising_facts_identified": ["facts with context and practical relevance"],
"contradictions_found": [
{{
"contradiction": "description of conflicting information",
"sections_involved": ["current", "related_section_X"],
"resolution_approach": "how addressed in podcast"
}}
],
"cross_section_connections": [
{{
"pattern": "connection description",
"sections_connected": ["section references"],
"actionable_application": "practical use of this connection"
}}
],
"actionable_takeaways": ["comprehensive recommendations with implementation steps"],
"value_delivery_moments": ["high-impact segments that provide maximum user benefit"]
}}
}}
Dynamic Quality Checklist:

 Primary focus on current_section with natural integration of related_sections
 Dialogue length varies strategically (short for transitions, long for key insights)
 Includes natural speech patterns ("ah," "um," "oh wow," "actually," "you know")
 Features authentic interruptions and overlapping thoughts ("hold on-" "but-" "exactly!")
 Each exchange serves a clear purpose (energy, depth, or practical value)
 Important concepts from current section get adequate explanation time
 Cross-section connections are explored meaningfully
 Contradictions between sections are addressed constructively
 Incorporates 2-3 key insights with appropriate depth
 Includes surprising facts with practical relevance
 Natural thinking moments and speech fillers enhance authenticity
 Addresses user needs with actionable guidance
 Maintains dynamic pacing throughout 2-5 minute duration
 Provides comprehensive takeaways in closing
 References source material naturally and credibly
 Delivers maximum practical value to listeners
 Creates multiple "aha" moments and actionable insights
 Ensures listeners leave with clear next steps and valuable knowledge

Note: The podcast should feel like a natural conversation between Alex and Emma as they explore the current section topic while weaving in insights from related sections. Balance dynamic pacing with deep value delivery and maintain authentic speech patterns throughout.
"""
