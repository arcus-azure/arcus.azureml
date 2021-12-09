const buildConfig = require('./docusaurus.config');

module.exports = {
  ...buildConfig,
  themeConfig: {
    ...buildConfig.themeConfig,
    algolia: {
      apiKey: process.env.ALGOLIA_API_KEY,
      indexName: 'arcus-azure',
      searchParameters: {
        facetFilters: ["tags:azure-ml"]
      },
    },
  }
}