# SubtitlesManagingModels.py

from datetime import datetime
# Removed unused imports like re, List, json, Optional, logging, Path

class DialogueLine:
    """Represents a single line of dialogue from an SRT file."""
    def __init__(self, line_number, start, end, text):
        try:
            self.line_number = int(line_number)
        except ValueError:
            # Handle cases where line_number might not be a simple integer
            # Log a warning or assign a default? For now, re-raise or log.
            raise ValueError(f"Invalid line number format: {line_number}")

        # Convert timecodes to seconds (float)
        self.start = self._timecode_to_sec(start)
        self.end = self._timecode_to_sec(end)
        self.text = str(text) # Ensure text is a string

    def to_dict(self):
        """Converts the DialogueLine object to a dictionary."""
        return {
            "line_number": self.line_number,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    def to_dict_only_number_and_text(self):
        """Converts to a dictionary containing only line number and text."""
        return {
            "line_number": self.line_number,
            "text": self.text,
        }

    @staticmethod
    def _timecode_to_sec(timecode):
        """Converts an SRT timecode string (or float) to seconds (float)."""
        if isinstance(timecode, (float, int)):
            return float(timecode)
        if not isinstance(timecode, str):
             raise TypeError(f"Timecode must be a string or float, got {type(timecode)}")

        timecode = timecode.strip()
        # Handle common variations
        timecode = timecode.replace('.', ',', 1) # Ensure comma separator for microseconds

        try:
            # Primary format H:M:S,ms
            dt = datetime.strptime(timecode, "%H:%M:%S,%f")
            return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
        except ValueError:
             # Fallback for potential missing milliseconds or other slight variations
            try:
                parts = timecode.split(':')
                if len(parts) == 3:
                    seconds_parts = parts[2].split(',')
                    seconds = float(seconds_parts[0])
                    milliseconds = float('0.' + seconds_parts[1]) if len(seconds_parts) > 1 else 0.0
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + seconds + milliseconds
                else:
                     raise ValueError(f"Cannot parse timecode format: {timecode}") # Re-raise if manual split fails
            except Exception as e:
                 # Catch any unexpected error during manual parsing
                 raise ValueError(f"Failed to parse timecode '{timecode}': {e}") from e


    def __str__(self):
        # String representation for easy debugging and logging.
        return (f"Line: {self.line_number}, "
                f"Time: {self.start:.3f} --> {self.end:.3f}, "
                f"Text: '{self.text[:50]}...'") # Show only partial text

    def __repr__(self):
         return f"DialogueLine(line_number={self.line_number}, start={self.start}, end={self.end}, text='{self.text[:20]}...')"