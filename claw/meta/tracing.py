"""Tracing and event logging for meta-cognitive controller."""

import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import asdict
from claw.types import Plan, TraceEvent, PlanStatus
from claw.logging import get_logger

logger = get_logger(__name__)


class TraceLogger:
    """Logger for planning trace events."""
    
    def __init__(self, trace_file: Optional[str] = None):
        """Initialize trace logger.
        
        Args:
            trace_file: Optional trace file path
        """
        self.trace_file = trace_file
        self.events: List[TraceEvent] = []
        
        if self.trace_file:
            # Ensure directory exists
            Path(self.trace_file).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Trace logging to: {self.trace_file}")
        else:
            logger.info("Trace logging disabled")
    
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a trace event.
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata
        """
        event = TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            metadata=metadata
        )
        
        self.events.append(event)
        
        # Write to file if enabled
        if self.trace_file:
            self._write_event_to_file(event)
        
        logger.debug(f"Logged event: {event_type}")
    
    def log_plan_generation(
        self,
        plans: List[Plan],
        method: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Log plan generation event.
        
        Args:
            plans: Generated plans
            method: Generation method used
            parameters: Generation parameters
        """
        plan_data = []
        for plan in plans:
            plan_data.append({
                'orders': plan.orders,
                'rationale': plan.rationale,
                'confidence': plan.confidence,
                'status': plan.status.value
            })
        
        self.log_event(
            event_type='plan_generation',
            data={
                'method': method,
                'plans': plan_data,
                'parameters': parameters
            }
        )
    
    def log_simulation(
        self,
        plan: Plan,
        results: List[Dict[str, Any]],
        opponent_model: str,
        samples: int
    ) -> None:
        """Log simulation event.
        
        Args:
            plan: Plan that was simulated
            results: Simulation results
            opponent_model: Opponent model used
            samples: Number of samples
        """
        self.log_event(
            event_type='simulation',
            data={
                'plan': {
                    'orders': plan.orders,
                    'rationale': plan.rationale,
                    'confidence': plan.confidence
                },
                'results': results,
                'opponent_model': opponent_model,
                'samples': samples
            }
        )
    
    def log_decision(
        self,
        decision: str,
        reason: str,
        context: Dict[str, Any]
    ) -> None:
        """Log decision event.
        
        Args:
            decision: Decision made
            reason: Reason for decision
            context: Decision context
        """
        self.log_event(
            event_type='decision',
            data={
                'decision': decision,
                'reason': reason,
                'context': context
            }
        )
    
    def log_repair(
        self,
        original_plan: Plan,
        repaired_plan: Optional[Plan],
        success: bool,
        feedback: Dict[str, Any]
    ) -> None:
        """Log plan repair event.
        
        Args:
            original_plan: Original plan
            repaired_plan: Repaired plan (if successful)
            success: Whether repair was successful
            feedback: Repair feedback
        """
        data = {
            'original_plan': {
                'orders': original_plan.orders,
                'rationale': original_plan.rationale,
                'confidence': original_plan.confidence
            },
            'success': success,
            'feedback': feedback
        }
        
        if repaired_plan:
            data['repaired_plan'] = {
                'orders': repaired_plan.orders,
                'rationale': repaired_plan.rationale,
                'confidence': repaired_plan.confidence
            }
        
        self.log_event(
            event_type='repair',
            data=data
        )
    
    def log_meta_decision(
        self,
        choice: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Log meta-policy decision.
        
        Args:
            choice: Choice made (llm, search, repair, stop)
            parameters: Parameters chosen
            context: Decision context
        """
        self.log_event(
            event_type='meta_decision',
            data={
                'choice': choice,
                'parameters': parameters,
                'context': context
            }
        )
    
    def log_iteration(
        self,
        iteration: int,
        best_plan: Optional[Plan],
        best_score: float,
        time_remaining: float,
        improvement: float
    ) -> None:
        """Log iteration summary.
        
        Args:
            iteration: Iteration number
            best_plan: Best plan found so far
            best_score: Best score so far
            time_remaining: Time remaining
            improvement: Improvement in this iteration
        """
        data = {
            'iteration': iteration,
            'best_score': best_score,
            'time_remaining': time_remaining,
            'improvement': improvement
        }
        
        if best_plan:
            data['best_plan'] = {
                'orders': best_plan.orders,
                'rationale': best_plan.rationale,
                'confidence': best_plan.confidence,
                'status': best_plan.status.value
            }
        
        self.log_event(
            event_type='iteration_summary',
            data=data
        )
    
    def log_error(
        self,
        error: str,
        context: Dict[str, Any]
    ) -> None:
        """Log error event.
        
        Args:
            error: Error message
            context: Error context
        """
        self.log_event(
            event_type='error',
            data={
                'error': error,
                'context': context
            }
        )
    
    def _write_event_to_file(self, event: TraceEvent) -> None:
        """Write event to trace file.
        
        Args:
            event: Event to write
        """
        try:
            with open(self.trace_file, 'a') as f:
                json.dump(asdict(event), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")
    
    def get_events(self, event_type: Optional[str] = None) -> List[TraceEvent]:
        """Get logged events.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            List of events
        """
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events.copy()
    
    def get_events_by_time_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[TraceEvent]:
        """Get events within time range.
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            List of events in time range
        """
        return [
            e for e in self.events
            if start_time <= e.timestamp <= end_time
        ]
    
    def clear_events(self) -> None:
        """Clear all logged events."""
        self.events.clear()
        logger.info("Cleared all trace events")
    
    def export_trace(self, filepath: str) -> None:
        """Export trace to file.
        
        Args:
            filepath: Path to export file
        """
        try:
            with open(filepath, 'w') as f:
                for event in self.events:
                    json.dump(asdict(event), f)
                    f.write('\n')
            logger.info(f"Exported trace to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")
    
    def load_trace(self, filepath: str) -> None:
        """Load trace from file.
        
        Args:
            filepath: Path to trace file
        """
        try:
            self.events.clear()
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        event_data = json.loads(line)
                        event = TraceEvent(**event_data)
                        self.events.append(event)
            logger.info(f"Loaded trace from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load trace: {e}")
