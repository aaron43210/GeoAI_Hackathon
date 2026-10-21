import React from 'react';

/** Minimal, safe markdown for the assistant's answers: bold, italics, bullets, paragraphs, [E#] citation chips. */
interface Props {
  text: string;
  onCite?: (id: string) => void;
}

function inline(s: string, onCite?: (id: string) => void, keyBase = ''): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|_[^_]+_|\*[^*]+\*|\[E\d+\])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) parts.push(s.slice(last, m.index));
    const tok = m[0];
    const key = `${keyBase}-${i++}`;
    if (tok.startsWith('**')) parts.push(<strong key={key}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith('[E')) {
      const id = tok.slice(1, -1);
      parts.push(
        <button key={key} className="cite" onClick={() => onCite?.(id)} title={`Show evidence ${id}`}>
          {id}
        </button>,
      );
    } else parts.push(<em key={key}>{tok.slice(1, -1)}</em>);
    last = m.index + tok.length;
  }
  if (last < s.length) parts.push(s.slice(last));
  return parts;
}

export const Markdown: React.FC<Props> = ({ text, onCite }) => {
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="md">
      {blocks.map((b, bi) => {
        const lines = b.split('\n').filter((l) => l.trim());
        const bullets = lines.filter((l) => /^\s*[-*•]\s+/.test(l));
        if (bullets.length && bullets.length === lines.length - (lines[0] && !/^\s*[-*•]/.test(lines[0]) ? 1 : 0)) {
          const head = !/^\s*[-*•]/.test(lines[0]) ? lines[0] : null;
          return (
            <div key={bi}>
              {head && <p>{inline(head, onCite, `h${bi}`)}</p>}
              <ul>
                {bullets.map((l, li) => (
                  <li key={li}>{inline(l.replace(/^\s*[-*•]\s+/, ''), onCite, `${bi}-${li}`)}</li>
                ))}
              </ul>
            </div>
          );
        }
        return <p key={bi}>{lines.map((l, li) => <React.Fragment key={li}>{li > 0 && <br />}{inline(l, onCite, `${bi}-${li}`)}</React.Fragment>)}</p>;
      })}
    </div>
  );
};
