import re
from typing import List, Dict, Any
from collections import defaultdict
import math

class EnhancedSearch:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Combine semantic search with keyword matching for better context awareness
        """
        # Get semantic results (expanded set)
        semantic_results = self.vector_db.search_similar(query, top_k=top_k * 2)
        
        # Perform keyword analysis
        query_keywords = self._extract_keywords(query)
        
        # Enhance results with keyword scores
        enhanced_results = []
        for result in semantic_results:
            # Calculate keyword overlap score
            keyword_score = self._calculate_keyword_score(result['text'], query_keywords)
            
            # Calculate structure relevance
            structure_score = self._calculate_structure_relevance(result, query)
            
            # Calculate final hybrid score
            hybrid_score = (
                0.6 * result['similarity_score'] +  # Semantic similarity (60%)
                0.25 * keyword_score +              # Keyword relevance (25%)
                0.15 * structure_score               # Structure relevance (15%)
            )
            
            enhanced_result = {
                **result,
                'hybrid_score': hybrid_score,
                'keyword_score': keyword_score,
                'structure_score': structure_score,
                'matching_keywords': self._find_matching_keywords(result['text'], query_keywords),
                'relevance_context': f"Hybrid search (Semantic: {result['similarity_score']:.3f}, Keywords: {keyword_score:.3f}, Structure: {structure_score:.3f})"
            }
            enhanced_results.append(enhanced_result)
        
        # Sort by hybrid score and return top results
        enhanced_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return enhanced_results[:top_k]
    
    def contextual_window_search(self, query: str, top_k: int = 10, window_size: int = 1) -> List[Dict[str, Any]]:
        """
        Expand search results with surrounding context for better understanding
        """
        base_results = self.hybrid_search(query, top_k)
        contextual_results = []
        
        for result in base_results:
            # Get surrounding chunks
            context = self._get_contextual_window(result, window_size)
            
            enhanced_result = {
                **result,
                'context_before': context['before'],
                'context_after': context['after'],
                'extended_text': context['combined'],
                'context_summary': self._summarize_context(context)
            }
            contextual_results.append(enhanced_result)
        
        return contextual_results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate how well the text matches the query keywords"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword in text_lower:
                # Count frequency and give bonus for exact matches
                frequency = text_lower.count(keyword)
                matches += frequency
        
        # Normalize by number of keywords
        return min(matches / len(keywords), 1.0)
    
    def _calculate_structure_relevance(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance based on document structure (headings, position)"""
        score = 0.0
        
        # Boost if heading contains query terms
        heading = result.get('heading', '').lower()
        query_lower = query.lower()
        
        if heading and any(term in heading for term in query_lower.split()):
            score += 0.5
        
        # Boost if it's from a prominent section (e.g., starts with numbers, contains "chapter", etc.)
        if heading and re.match(r'^\d+\.', heading):
            score += 0.2
        
        # Boost based on document type or importance indicators
        doc_name = result.get('doc', '').lower()
        if any(term in doc_name for term in ['guide', 'manual', 'tutorial', 'reference']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _find_matching_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords are present in the text"""
        text_lower = text.lower()
        return [keyword for keyword in keywords if keyword in text_lower]
    
    def _get_contextual_window(self, result: Dict[str, Any], window_size: int) -> Dict[str, str]:
        """Get surrounding context chunks for better understanding"""
        # This would need to be implemented based on your chunk storage system
        # For now, return the original text as context
        return {
            'before': '',  # Would contain previous chunks
            'after': '',   # Would contain next chunks
            'combined': result['text']  # Original text for now
        }
    
    def _summarize_context(self, context: Dict[str, str]) -> str:
        """Provide a brief summary of the contextual information"""
        combined_length = len(context['combined'])
        
        if combined_length > 500:
            return f"Extended context available ({combined_length} characters)"
        else:
            return "Standard context"
    
    def intent_aware_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze query intent and adjust search strategy accordingly with deduplication
        """
        intent = self._analyze_query_intent(query)
        
        # Get base results (larger set for better deduplication)
        results = self.hybrid_search(query, top_k * 3)
        
        # Filter and rank based on intent
        filtered_results = []
        for result in results:
            intent_score = self._score_by_intent(result, intent)
            
            # Update relevance context with intent information
            base_context = result.get('relevance_context', f"Semantic: {result['similarity_score']:.3f}")
            intent_context = f"{base_context}, Intent: {intent}({intent_score:.2f})"
            
            enhanced_result = {
                **result,
                'intent': intent,
                'intent_score': intent_score,
                'final_score': result['hybrid_score'] * (1 + intent_score * 0.3),
                'relevance_context': intent_context
            }
            filtered_results.append(enhanced_result)
        
        # Sort by final score
        filtered_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Deduplicate results by section/heading
        deduplicated_results = self._deduplicate_by_section(filtered_results)
        
        # Apply diversity filter to ensure variety in results
        diverse_results = self._apply_diversity_filter(deduplicated_results, top_k)
        
        return diverse_results
    
    def _apply_diversity_filter(self, results: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """
        Ensure diversity in results by preferring different documents and section types
        """
        if len(results) <= target_count:
            return results
        
        diverse_results = []
        document_counts = defaultdict(int)
        section_types = set()
        
        # First pass: include top results while maintaining diversity
        for result in results:
            doc = result.get('doc', '')
            heading = result.get('heading', '').lower()
            
            # Determine section type based on heading
            section_type = self._classify_section_type(heading)
            
            # Add result if we haven't exceeded document limit or it's a new section type
            doc_limit = max(1, target_count // 3)  # Max 1/3 of results from same document
            
            if (document_counts[doc] < doc_limit or 
                section_type not in section_types or 
                len(diverse_results) < target_count // 2):
                
                diverse_results.append(result)
                document_counts[doc] += 1
                section_types.add(section_type)
                
                if len(diverse_results) >= target_count:
                    break
        
        # Second pass: fill remaining slots with best remaining results
        remaining_slots = target_count - len(diverse_results)
        if remaining_slots > 0:
            remaining_results = [r for r in results if r not in diverse_results]
            diverse_results.extend(remaining_results[:remaining_slots])
        
        return diverse_results
    
    def _classify_section_type(self, heading: str) -> str:
        """
        Classify section type based on heading content
        """
        if not heading:
            return 'general'
        
        heading_lower = heading.lower()
        
        if any(term in heading_lower for term in ['introduction', 'overview', 'getting started']):
            return 'introduction'
        elif any(term in heading_lower for term in ['tutorial', 'how to', 'guide', 'step']):
            return 'tutorial'
        elif any(term in heading_lower for term in ['reference', 'api', 'function', 'command']):
            return 'reference'
        elif any(term in heading_lower for term in ['example', 'sample', 'demo']):
            return 'example'
        elif any(term in heading_lower for term in ['troubleshoot', 'error', 'problem', 'fix']):
            return 'troubleshooting'
        elif any(term in heading_lower for term in ['advanced', 'expert', 'pro']):
            return 'advanced'
        else:
            return 'general'
    
    def _deduplicate_by_section(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results from the same section/heading, keeping the best one
        """
        section_groups = defaultdict(list)
        
        # Group results by document + section combination
        for result in results:
            doc = result.get('doc', '')
            heading = result.get('heading', '')
            page = result.get('page', 0)
            
            # Create a unique key for this section
            # Use document + heading, or document + page if no heading
            if heading and heading.strip():
                section_key = f"{doc}::{heading.strip()}"
            else:
                section_key = f"{doc}::page_{page}"
            
            section_groups[section_key].append(result)
        
        # Keep only the best result from each section
        deduplicated_results = []
        for section_key, group in section_groups.items():
            if len(group) == 1:
                # Only one result for this section, keep it
                result = group[0]
                # Ensure required fields are present
                if 'relevance_context' not in result:
                    result['relevance_context'] = f"Semantic: {result.get('similarity_score', 0):.3f}"
                deduplicated_results.append(result)
            else:
                # Multiple results for same section, keep the best one and merge others
                best_result = max(group, key=lambda x: x.get('final_score', x.get('hybrid_score', x.get('similarity_score', 0))))
                
                # Merge content from other chunks in the same section
                merged_content = self._merge_section_content(group, best_result)
                best_result['text'] = merged_content
                best_result['merged_chunks'] = len(group)
                
                # Update relevance context to show merging
                base_context = best_result.get('relevance_context', f"Semantic: {best_result.get('similarity_score', 0):.3f}")
                best_result['relevance_context'] = f"{base_context} (Merged {len(group)} chunks from same section)"
                
                deduplicated_results.append(best_result)
        
        # Sort again after deduplication
        deduplicated_results.sort(key=lambda x: x.get('final_score', x.get('hybrid_score', x.get('similarity_score', 0))), reverse=True)
        return deduplicated_results
    
    def _merge_section_content(self, chunks: List[Dict[str, Any]], best_chunk: Dict[str, Any]) -> str:
        """
        Intelligently merge content from multiple chunks in the same section
        """
        # Sort chunks by page number and position if available
        sorted_chunks = sorted(chunks, key=lambda x: (x.get('page', 0), x.get('position', 0)))
        
        # Combine text content, removing duplicates
        combined_text = []
        seen_sentences = set()
        
        for chunk in sorted_chunks:
            text = chunk['text'].strip()
            
            # Split into sentences for deduplication
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen_sentences and len(sentence) > 20:
                    combined_text.append(sentence)
                    seen_sentences.add(sentence)
        
        # Join sentences and limit length
        merged_text = '. '.join(combined_text)
        
        # If merged text is too long, prioritize content from the best chunk
        if len(merged_text) > 1000:
            best_text = best_chunk['text']
            # Take best chunk text + some additional context
            additional_context = merged_text.replace(best_text, '').strip()
            if additional_context:
                merged_text = best_text + '\n\n[Additional context: ' + additional_context[:200] + '...]'
            else:
                merged_text = best_text
        
        return merged_text
    
    def _analyze_query_intent(self, query: str) -> str:
        """Determine the type of information the user is seeking"""
        query_lower = query.lower()
        
        # Definition patterns
        if any(term in query_lower for term in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        
        # Procedure patterns
        if any(term in query_lower for term in ['how to', 'steps', 'process', 'procedure', 'guide']):
            return 'procedure'
        
        # Example patterns
        if any(term in query_lower for term in ['example', 'sample', 'instance', 'case']):
            return 'example'
        
        # Comparison patterns
        if any(term in query_lower for term in ['vs', 'versus', 'compare', 'difference', 'better']):
            return 'comparison'
        
        # Troubleshooting patterns
        if any(term in query_lower for term in ['error', 'problem', 'issue', 'fix', 'solve', 'troubleshoot']):
            return 'troubleshooting'
        
        return 'general'
    
    def _score_by_intent(self, result: Dict[str, Any], intent: str) -> float:
        """Score results based on how well they match the query intent"""
        text_lower = result['text'].lower()
        
        if intent == 'definition':
            # Look for definitional language
            if any(term in text_lower for term in ['is defined as', 'refers to', 'means', 'is a type of']):
                return 0.8
            if any(term in text_lower for term in [' is ', ' are ', 'definition']):
                return 0.5
        
        elif intent == 'procedure':
            # Look for procedural language
            if any(term in text_lower for term in ['step', 'first', 'then', 'next', 'finally']):
                return 0.8
            if re.search(r'\d+\.', text_lower):  # Numbered lists
                return 0.7
        
        elif intent == 'example':
            # Look for example language
            if any(term in text_lower for term in ['example', 'for instance', 'such as', 'like']):
                return 0.8
            if 'scenario' in text_lower:
                return 0.6
        
        elif intent == 'troubleshooting':
            # Look for problem-solving language
            if any(term in text_lower for term in ['solution', 'fix', 'resolve', 'error', 'problem']):
                return 0.8
        
        return 0.0
