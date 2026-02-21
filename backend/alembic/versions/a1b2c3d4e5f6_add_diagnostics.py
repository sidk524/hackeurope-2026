"""Add diagnostics tables

Revision ID: a1b2c3d4e5f6
Revises: 5ff8b868ffb1
Create Date: 2026-02-21 18:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '5ff8b868ffb1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'diagnosticrun',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('health_score', sa.Integer(), nullable=False),
        sa.Column('issue_count', sa.Integer(), nullable=False),
        sa.Column('arch_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('summary_json', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['trainsession.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'diagnosticissue',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('severity', sa.Enum('critical', 'warning', 'info', name='issueseverity'), nullable=False),
        sa.Column('category', sa.Enum('loss', 'throughput', 'memory', 'profiler', 'logs', 'system', 'architecture', 'sustainability', name='issuecategory'), nullable=False),
        sa.Column('title', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('description', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('epoch_index', sa.Integer(), nullable=True),
        sa.Column('layer_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('metric_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('metric_value', sa.JSON(), nullable=True),
        sa.Column('suggestion', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['diagnosticrun.id']),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('diagnosticissue')
    op.drop_table('diagnosticrun')
    # Drop enums explicitly for non-SQLite backends
    sa.Enum(name='issueseverity').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='issuecategory').drop(op.get_bind(), checkfirst=True)
