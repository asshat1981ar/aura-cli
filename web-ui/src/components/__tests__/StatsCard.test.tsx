import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { StatsCard } from '../StatsCard'
import { Target } from 'lucide-react'

describe('StatsCard', () => {
  it('should render title and value', () => {
    render(
      <StatsCard
        title="Test Title"
        value="100"
        description="Test description"
        icon={Target}
      />
    )

    expect(screen.getByText('Test Title')).toBeInTheDocument()
    expect(screen.getByText('100')).toBeInTheDocument()
    expect(screen.getByText('Test description')).toBeInTheDocument()
  })

  it('should render trend indicator when provided', () => {
    render(
      <StatsCard
        title="Test"
        value="100"
        icon={Target}
        trend={{ value: 5.2, isPositive: true }}
      />
    )

    expect(screen.getByText('+5.2%')).toBeInTheDocument()
  })

  it('should render negative trend correctly', () => {
    render(
      <StatsCard
        title="Test"
        value="100"
        icon={Target}
        trend={{ value: 3.5, isPositive: false }}
      />
    )

    expect(screen.getByText('-3.5%')).toBeInTheDocument()
  })
})
