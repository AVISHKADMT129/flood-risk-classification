export default function ExplanationText({ explanation, riskClass }) {
  if (!explanation) return null;

  return (
    <div className={`explanation-section explanation-${riskClass}`}>
      <p className="explanation-text">{explanation}</p>
    </div>
  );
}
